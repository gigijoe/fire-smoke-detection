// Standard include files
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/cudacodec.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <errno.h>
#include <fcntl.h> 
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#define MAX_UCHAR (0xFF)

using namespace cv;
using namespace std;

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

//#define JETSON_NANO

#define CAMERA_WIDTH 640
#define CAMERA_HEIGHT 480
#define CAMERA_FPS 60

#define VIDEO_INPUT_FILE "../video/fire1.avi"
//#define VIDEO_INPUT_FILE "../video/fBackYardFire.avi"
//#define VIDEO_INPUT_FILE "../video/test_fire_1.mp4"
//#define VIDEO_INPUT_FILE "../video/test_fire_2.mp4"
//#define VIDEO_INPUT_FILE "../video/fire_smoke.mp4"

#define VIDEO_OUTPUT_SCREEN

#define VIDEO_OUTPUT_DIR "."
#define VIDEO_OUTPUT_FILE_NAME "fire"

#ifdef VIDEO_OUTPUT_FILE_NAME
#ifndef VIDEO_OUTPUT_SCREEN
#define VIDEO_OUTPUT_SCREEN
#endif 
#endif

#define VIDEO_FRAME_DROP 30

#define F3F

#ifdef JETSON_NANO
#define F3F_TTY_BASE
#endif

static bool bShutdown = false;

void sig_handler(int signo)
{
    if(signo == SIGINT) {
        printf("SIGINT\n");
        bShutdown = true;
    }
}

/*
*
*/

static int set_interface_attribs (int fd, int speed, int parity)
{
    struct termios tty;
    memset (&tty, 0, sizeof tty);
    if (tcgetattr (fd, &tty) != 0)
    {
            printf ("error %d from tcgetattr\n", errno);
            return -1;
    }

    cfsetospeed (&tty, speed);
    cfsetispeed (&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
    // disable IGNBRK for mismatched speed tests; otherwise receive break
    // as \000 chars
    tty.c_iflag &= ~IGNBRK;         // disable break processing
    tty.c_lflag = 0;                // no signaling chars, no echo,
                                    // no canonical processing
    tty.c_oflag = 0;                // no remapping, no delays
    tty.c_cc[VMIN]  = 0;            // read doesn't block
    tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

    tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
                                    // enable reading
    tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
    tty.c_cflag |= parity;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr (fd, TCSANOW, &tty) != 0)
    {
            printf ("error %d from tcsetattr\n", errno);
            return -1;
    }
    return 0;
}

static void set_blocking (int fd, int should_block)
{
    struct termios tty;
    memset (&tty, 0, sizeof tty);
    if (tcgetattr (fd, &tty) != 0)
    {
            printf ("error %d from tggetattr\n", errno);
            return;
    }

    tty.c_cc[VMIN]  = should_block ? 1 : 0;
    tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

    if (tcsetattr (fd, TCSANOW, &tty) != 0)
            printf ("error %d setting term attributes\n", errno);
}

#define BASE_A

static void base_toggle (int fd) 
{
    static uint8_t serNo = 0;
    uint8_t data[1];

    if(!fd)
        return;

#ifdef BASE_A
    data[0] = (serNo & 0x3f);
    printf("BASE_A[%d]\r\n", serNo);
#endif    
#ifdef BASE_B
    data[0] = (serNo & 0x3f) | 0x40;
    printf("BASE_B[%d]\r\n", serNo);
#endif
    write(fd, data, 1);

    if(++serNo > 0x3f)
        serNo = 0;
}

/*
*
*/

static inline bool ContoursSort(vector<cv::Point> contour1, vector<cv::Point> contour2)  
{  
    //return (contour1.size() > contour2.size()); /* Outline length */
    return (cv::contourArea(contour1) > cv::contourArea(contour2)); /* Area */
}  

inline void writeText( Mat & mat, const string text )
{
   int fontFace = FONT_HERSHEY_SIMPLEX;
   double fontScale = 1;
   int thickness = 1;  
   Point textOrg( 10, 40 );
   putText( mat, text, textOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness, cv::LINE_8 );
}

class FrameQueue
{
public:
    struct cancelled {};

public:
    FrameQueue() : cancelled_(false) {}

    void push(Mat const & image);
    Mat pop();

    void cancel();
    bool isCancelled() { return cancelled_; }

private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool cancelled_;
};

void FrameQueue::cancel()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    cancelled_ = true;
    cond_.notify_all();
}

void FrameQueue::push(cv::Mat const & image)
{
    while(queue_.size() > 30) { /* Prevent memory overflow ... */
        usleep(1000); /* Wait for 1 ms */
    }

    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(image);
    cond_.notify_one();
}

Mat FrameQueue::pop()
{
    std::unique_lock<std::mutex> mlock(mutex_);

    while (queue_.empty()) {
        if (cancelled_) {
            throw cancelled();
        }
        cond_.wait(mlock);
        if (cancelled_) {
            throw cancelled();
        }
    }

    Mat image(queue_.front());
    queue_.pop();
    return image;
}

#if defined(VIDEO_OUTPUT_FILE_NAME)
FrameQueue videoWriterQueue;

void VideoWriterThread(int width, int height)
{    
    Size videoSize = Size((int)width,(int)height);
    VideoWriter writer;
    char filePath[64];
    int videoOutoutIndex = 0;
    while(videoOutoutIndex < 1000) {
        snprintf(filePath, 64, "%s/%s%03d.mp4", VIDEO_OUTPUT_DIR, VIDEO_OUTPUT_FILE_NAME, videoOutoutIndex);
        FILE *fp = fopen(filePath, "rb");
        if(fp) { /* file exist ... */
            fclose(fp);
            videoOutoutIndex++;
        } else
            break; /* File doesn't exist. OK */
    }
#ifdef JETSON_NANO
    char gstStr[256];
    snprintf(gstStr, 256, "appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=%s/%s%03d.mkv ", 
        VIDEO_OUTPUT_DIR, VIDEO_OUTPUT_FILE_NAME, videoOutoutIndex);
    writer.open(gstStr, VideoWriter::fourcc('X', '2', '6', '4'), 30, videoSize);
    cout << "Vodeo output " << gstStr << endl;
#else
    writer.open(filePath, VideoWriter::fourcc('X', '2', '6', '4'), 30, videoSize);
    cout << "Vodeo output " << filePath << endl;
#endif
    try {
        while(1) {
            Mat frame = videoWriterQueue.pop();
            writer.write(frame);
        }
    } catch (FrameQueue::cancelled & /*e*/) {
        // Nothing more to process, we're done
        std::cout << "FrameQueue " << " cancelled, worker finished." << std::endl;
    }    
}

#endif

int main(int argc, char**argv)
{
    double fps = 0;
#ifdef F3F_TTY_BASE
    const char *ttyName = "/dev/ttyTHS1";

    int ttyFd = open (ttyName, O_RDWR | O_NOCTTY | O_SYNC);
    if (ttyFd) {
        set_interface_attribs (ttyFd, B9600, 0);  // set speed to 115,200 bps, 8n1 (no parity)
        set_blocking (ttyFd, 0);                // set no blocking
    } else
        printf ("error %d opening %s: %s\n", errno, ttyName, strerror (errno));
#endif

    if(signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    Mat frame, capFrame;
    cuda::GpuMat gpuFrame;

    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    std::cout << cv::getBuildInformation() << std::endl;

#ifdef JETSON_NANO
    static char gstStr[320];
#endif
#ifdef VIDEO_INPUT_FILE
#ifdef JETSON_NANO
    snprintf(gstStr, 320, "filesrc location=%s ! qtdemux name=demux demux.video_0 ! queue ! "
        "h264parse ! omxh264dec ! videoconvert ! appsink ", VIDEO_INPUT_FILE);
    VideoCapture cap(gstStr, cv::CAP_GSTREAMER);
#else
    VideoCapture cap(VIDEO_INPUT_FILE, cv::CAP_FFMPEG);
#endif
#else
    int index = 0;    
    if(argc > 1)
        index = atoi(argv[1]);
#ifdef JETSON_NANO
    /* export GST_DEBUG=2 to show debug message */
    snprintf(gstStr, 320, "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! \
        nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink -e", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS);

    VideoCapture cap(gstStr, cv::CAP_GSTREAMER);
#else
    VideoCapture cap(index);
#endif

#ifdef JETSON_NANO
        cout << "Video input " << gstStr << endl;
#else
        cout << "Video input (" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_HEIGHT))
            << ") at " << cap.get(CAP_PROP_FPS) << " FPS." << endl;
#endif

#endif
    if(!cap.isOpened()) {
        cout << "Could not open video" << endl;
        return 1;
    }

#ifdef VIDEO_INPUT_FILE
    cap.read(capFrame);
#else
#ifdef JETSON_NANO
    cout << "Video input " << gstStr << endl;
#else
    cap.set(CAP_PROP_FOURCC ,VideoWriter::fourcc('M', 'J', 'P', 'G') );
    cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(CAP_PROP_FPS, 30.0);

    cout << "Video input (" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_HEIGHT))
        << ") at " << cap.get(CAP_PROP_FPS) << " FPS." << endl;
#endif
    cout << "Drop first " << VIDEO_FRAME_DROP << " for camera stable ..." << endl;
    for(int i=0;i<VIDEO_FRAME_DROP;i++) {
        if(!cap.read(frame))
            printf("Error read camera frame ...\n");
    }
#endif

#if defined(VIDEO_OUTPUT_FILE_NAME)
    thread outThread(&VideoWriterThread, capFrame.cols, capFrame.rows);
#endif
    //Ptr<BackgroundSubtractor> bsModel = createBackgroundSubtractorKNN();
    //Ptr<BackgroundSubtractor> bsModel = createBackgroundSubtractorMOG2();
    /* 30 : history, 16 : threshold */
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel = cuda::createBackgroundSubtractorMOG2(32, 16, false);

    bool doUpdateModel = true;
    bool doSmoothMask = true;

    Mat foregroundMask, background;
#ifdef VIDEO_OUTPUT_SCREEN
    Mat outFrame;
#endif
    cuda::GpuMat gpuForegroundMask;

    high_resolution_clock::time_point t1(high_resolution_clock::now());

    while(cap.read(capFrame)) {
        //cvtColor(capFrame, frame, COLOR_BGR2GRAY);
        frame = capFrame;
#ifdef VIDEO_OUTPUT_SCREEN
        capFrame.copyTo(outFrame);
#endif //VIDEO_OUTPUT_SCREEN
        gpuFrame.upload(frame);

        // pass the frame to background bsModel
        bsModel->apply(gpuFrame, gpuForegroundMask, doUpdateModel ? -1 : 0);
#if 0
/*
    m = max{ R ( i , j ), G ( i , j ), B ( i , j )}
    n = min{ R ( i , j ), G ( i , j ), B ( i , j )}
    5 < m - n < 20
*/

        cuda::GpuMat gpuRgbCh[3];
        cuda::split(gpuFrame, gpuRgbCh);

        cuda::GpuMat gpuMaxCh;
        cuda::max(gpuRgbCh[0], gpuRgbCh[1], gpuMaxCh);
        cuda::max(gpuMaxCh, gpuRgbCh[2], gpuMaxCh);

        cuda::GpuMat gpuMinCh;
        cuda::min(gpuRgbCh[0], gpuRgbCh[1], gpuMinCh);
        cuda::min(gpuMinCh, gpuRgbCh[2], gpuMinCh);

        cuda::GpuMat gpuDiff;
        cuda::absdiff(gpuMaxCh, gpuMinCh, gpuDiff);

        cuda::GpuMat gpuA;
        cuda::GpuMat mat_high, mat_low;
        cuda::threshold(gpuDiff, mat_low, 5, MAX_UCHAR, THRESH_BINARY);
        cuda::threshold(gpuDiff, mat_high, 20, MAX_UCHAR, THRESH_BINARY_INV);
        cuda::bitwise_and(mat_high, mat_low, gpuA);

/*
    80 < L < 150 
    190 < L < 255 
*/

        cuda::GpuMat gpuHls;
        cuda::cvtColor(gpuFrame, gpuHls, COLOR_RGB2HLS);

        //Split HSL 3 channels
        cuda::GpuMat gpuHlsCh[3];
        cuda::split(gpuHls, gpuHlsCh);

        //cuda::GpuMat mat_high, mat_low;
        cuda::GpuMat thresc;

        //Threshold L channel
        cuda::threshold(gpuHlsCh[1], mat_low, 190, MAX_UCHAR, THRESH_BINARY);
        cuda::threshold(gpuHlsCh[1], mat_high, 255, MAX_UCHAR, THRESH_BINARY_INV);
        cuda::bitwise_and(mat_high, mat_low, thresc);

        cuda::GpuMat gpuResult;
        cuda::bitwise_and(gpuA, thresc, gpuResult);
        cuda::bitwise_and(gpuForegroundMask, gpuResult, gpuResult);

        gpuResult.download(foregroundMask);
#endif
#if 1
        cuda::GpuMat gpuHls;
        cuda::cvtColor(gpuFrame, gpuHls, COLOR_RGB2HLS);

        //Split HSL 3 channels
        cuda::GpuMat gpuHlsCh[3];
        cuda::split(gpuHls, gpuHlsCh);

        cuda::GpuMat mat_high, mat_low;
        cuda::GpuMat thresc[3];
        //Threshold H channel
        cuda::threshold(gpuHlsCh[0], mat_low, 85, MAX_UCHAR, THRESH_BINARY); /* 0 ~ 60 degree */
        cuda::threshold(gpuHlsCh[0], mat_high, 170, MAX_UCHAR, THRESH_BINARY_INV); /* 0 ~ 60 degree */
        cuda::bitwise_and(mat_high, mat_low, thresc[0]);

        //Threshold L channel
        cuda::threshold(gpuHlsCh[1], mat_low, 127, MAX_UCHAR, THRESH_BINARY); /* 127 ~ 255 */
        cuda::threshold(gpuHlsCh[1], mat_high, 235, MAX_UCHAR, THRESH_BINARY_INV); /* 127 ~ 255 */
        cuda::bitwise_and(mat_high, mat_low, thresc[1]);

        //Threshold S channel
        cuda::threshold(gpuHlsCh[2], mat_low, 0, MAX_UCHAR, THRESH_BINARY); /* 0 ~ 0.2 */
        cuda::threshold(gpuHlsCh[2], mat_high, 255, MAX_UCHAR, THRESH_BINARY_INV); /* 0 ~ 0.2 */
        cuda::bitwise_and(mat_high, mat_low, thresc[2]);

        cuda::GpuMat gpuTemp, gpuResult;
        //Bitwise AND the channels
        cuda::bitwise_and(thresc[0], thresc[1], gpuTemp);
        cuda::bitwise_and(gpuTemp, thresc[2], gpuResult);
        cuda::bitwise_and(gpuForegroundMask, gpuResult, gpuResult);

        gpuResult.download(foregroundMask);
#endif
#if 1
        int erosion_size = 1;   
        Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                          cv::Point(erosion_size, erosion_size) );
        erode(foregroundMask, foregroundMask, element);
#endif
        vector< vector<Point> > contours;
        vector< Vec4i > hierarchy;
        findContours(foregroundMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        vector<Rect> boundRect( contours.size() );
        Rect resultRect;
        for(int i=0; i<contours.size(); i++) {
            approxPolyDP( Mat(contours[i]), contours[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours[i]) );
            resultRect |= boundRect[i];
#ifdef VIDEO_OUTPUT_SCREEN
            Scalar color = Scalar( 0, 255, 0 );
            rectangle( outFrame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
#endif
        }
#ifdef VIDEO_OUTPUT_SCREEN
        Scalar color = Scalar( 0, 0, 255 );
        rectangle( outFrame, resultRect.tl(), resultRect.br(), color, 2, 8, 0 );
#endif
/*
        cvtColor(foregroundMask, foregroundMask, COLOR_GRAY2BGR);
        imshow("foreground mask", foregroundMask);
*/
/*
        bsModel->getBackgroundImage(background);
        if (!background.empty())
            imshow("mean background image", background );
*/
#ifdef VIDEO_OUTPUT_SCREEN
        char str[32];
        snprintf(str, 32, "FPS : %.2lf", fps);
        writeText(outFrame, string(str));

        imshow("Out Frame", outFrame);
#if defined(VIDEO_OUTPUT_FILE_NAME)
        videoWriterQueue.push(outFrame.clone());
        //videoWriterQueue.push(foregroundMask.clone());
#endif        
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        int k = waitKey(1);
        if(k == 27) {
            break;
        } else if(k == 'p') {
            while(waitKey(1) != 'p') {
                if(bShutdown)
                    break;
            }
        }
#endif
        if(bShutdown)
            break;

        while(1) {
            high_resolution_clock::time_point t2(high_resolution_clock::now());
            double dt_us(static_cast<double>(duration_cast<microseconds>(t2 - t1).count()));
            if(dt_us > 33000) /* 33 ms*/
                break;
            usleep(1000);
        }

        high_resolution_clock::time_point t2(high_resolution_clock::now());
        double dt_us(static_cast<double>(duration_cast<microseconds>(t2 - t1).count()));
        //std::cout << (dt_us / 1000.0) << " ms" << std::endl;
        fps = (1000000.0 / dt_us);
        std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << std::endl;

        t1 = high_resolution_clock::now();
    }

    //cap.release();

#if defined(VIDEO_OUTPUT_FILE_NAME)
    videoWriterQueue.cancel();
    outThread.join();
#endif

#ifdef F3F_TTY_BASE
    if(ttyFd)
        close(ttyFd);
#endif
    std::cout << "Finished ..." << endl;

    return 0;     
}
