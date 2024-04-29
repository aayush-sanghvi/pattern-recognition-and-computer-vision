/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 27st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To display different filters and display those filter
using the OpenCV library.
*/

//Include all the standard and necessary libraries.
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <opencv4/opencv2/opencv.hpp>
#include "filters.h"
#include "faceDetect.h"

//Main function
int main(int argc, char *argv[])
{
    //checks if image argument is mentioned otherwise exits the program with -1.
    if(argc<2)
    {
        printf("Please specify the image to display");
        exit(-1);
    }

    //declare variables to store each frame of video.
    cv::Mat frame;
    cv::Mat destination;
    cv::Mat liveVideo;

    //creates a window to display the video.
    cv::namedWindow(argv[1],cv::WINDOW_NORMAL);
    //cv::namedWindow("Live Video",cv::WINDOW_NORMAL);


    //reads the video and stores it in opencv VideoCapture datatype variable.
    cv::VideoCapture video((std::string)argv[1]);
    cv::VideoCapture cap(0);
    
    //checks if the video capture has been initialise and captured video is a valid video.
    if(!video.isOpened())
    {
        std::cout<<"Could not open video";
        exit(-1);
    }
    if(!cap.isOpened())
    {
        std::cout<<"Could not open video";
        exit(-1);
    }
        
    //creates an infinite loop.
    while(1)
    {
        //stores a new frame from the video into a variable defined above.
        video.read(frame);
        cap>>liveVideo;
        
        //checks for end of the video and then resets the video again from start to play the video in a loop.
        if(liveVideo.empty())
        {
            video.set(cv::CAP_PROP_POS_FRAMES,0);
            break;
        }

        //checks if filters are implemented, if not it will display the original video.
        if(destination.empty())
        {
            destination=liveVideo;
        }

        imageModification(liveVideo,destination);

        //displays the modified frame on the window created ealier
        cv::imshow(argv[1],destination);

        //checks for 'q' key press by the user to exit the program.
        if(cv::waitKey(1)=='q')
        {
            break;
        }
    }

    //releases the video and terminates the window created.
    video.release();
    liveVideo.release();
    cv::destroyAllWindows();

    //ends the main function by returning 0 value to main program.
    return 0;
}
