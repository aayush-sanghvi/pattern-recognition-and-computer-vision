/*
Aayush H Sanghvi and Yogi H Shah
Spring 2024 semseter
Date:- 18th Mar 2024
CS5330- Pattern Recognition and Computer Vision.

To calibrate the camera and then use the calibration 
to generate virtual objects in a scene.
*/

//Include all the standard and necessary libraries.
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "calibration.h"

//Main Function
int main(int argc, char *argv[]) {
    // declare all the necessary variables.
    cv::Size chessboardPatternSize(9, 6); // the size of the chessboard, height is 6, width is 9
    cv::Mat chessboardCameraMatrix; // matrix to store camera matrix.
    std::vector<cv::Vec3f> chessBoardPoints; // the 3D world points constructed for the chessboard target
    std::vector<std::vector<cv::Point2f>> chessboardCornerList; // vector to store chessboard corner list
    std::vector<std::vector<cv::Vec3f>> chessboardPointList; // vector to store chessboard point list
    cv::Mat chessboardDistCoeffs; // matrix to store cheesboard distance coefficients
    std::vector<cv::Mat> chessboardR, chessboardT; // vectors to store rotational and translational matrix of camera calibration
    bool chessboardCornersFlag=false;
    bool harrisCornerFlag=false;

    //declare variables to store each frame of video.
    cv::Mat frame;

    //creates a window to display the video.
    cv::namedWindow("Video",cv::WINDOW_NORMAL);

        // Save IP Address and port number for DroidCam application.
    std::string ipAddress = "10.0.0.52";
    int port = 4747;

    // URL for connecting to DroidCam
    std::string url = "http://" + ipAddress + ":" + std::to_string(port) + "/video";

    //reads the video and stores it in opencv VideoCapture datatype variable.
    cv::VideoCapture cap(0);

    //checks if the video capture has been initialise and captured video is a valid video.
    if(!cap.isOpened())
    {
        std::cout<<"Could not open video";
        exit(-1);
    }

    // create a vector of points that specifies the 3D position of the corners in world coordinates
    chessBoardPoints = worldCoordinates(chessboardPatternSize);    

    //creates an infinite loop.
    while (true) {
        //stores a new frame from the video into a variable defined above.
        cap >> frame;

        //checks for the frames in the video, if frames are empty it will break out of the loop.
        if (frame.empty()) {
            std::cout << "frame is empty\n";
            break;
        }

        // copy video frames into another variable to perform augmented reality and calibration.
        cv::Mat displayedFrame = frame.clone();

        // check if there is any key pressed by the user.
        char key = cv::waitKey(10);

        // extract chessboard corners and draw chessboard corners on live video.
        std::vector<cv::Point2f> chessboardCorners;
        if (chessBoardCorners(frame, chessboardPatternSize, chessboardCorners)) {
            chessboardCornersFlag=true;
            drawChessboardCorners(displayedFrame, chessboardPatternSize, chessboardCorners, true);
        }

        //if user pressed 's' key, select the calibration image for chessboard and add the corners and real-world points into a vector
        if (key == 's') {
            if (chessboardCornersFlag) {
                std::cout << "select chessboard calibration image" << std::endl;
                chessboardCornerList.push_back(chessboardCorners);
                chessboardPointList.push_back(chessBoardPoints);
            }
            else {
                std::cout << "No chessboard corners found" << std::endl;
            }
        }
        //if the user pressed 'c' key, calibrate the camera for chessboard.
        else if (key == 'c') {
            //select at-least 5 calibration image.
            if (chessboardPointList.size() < 5) {
                printf("Select atleast 5 or more calibration frames.\n");
            } 
            else {
                printf("calibrate camera\n");
                // calibrate the camera
                double chessboardError = calibrateCamera(chessboardPointList, chessboardCornerList, cv::Size(frame.rows, frame.cols), chessboardCameraMatrix, chessboardDistCoeffs, chessboardR, chessboardT);

                // print out the intrinsic parameters and the final re-projection error
                printf("Chessboard Camera Matrix: \n");
                for (int i = 0; i < chessboardCameraMatrix.rows; i++) {
                    for (int j = 0; j < chessboardCameraMatrix.cols; j++) {
                        std::cout << chessboardCameraMatrix.at<double>(i, j) << ", ";
                    }
                    printf("\n");
                }
                printf("\n");
                printf("Chessboard Distortion Coefficients: \n");
                for (int i = 0; i < chessboardDistCoeffs.rows; i++) {
                    for (int j = 0; j < chessboardDistCoeffs.cols; j++) {
                        std::cout << chessboardDistCoeffs.at<double>(i, j) << ", ";
                    }
                    printf("\n");
                }
                printf("\n");
                std::cout << "Chessboard Re-projection Error: " << chessboardError << std::endl;
                // printf("\n");
                // printf("Chessboard Rotational Matrix: \n");
                // for (const auto& innerArray : chessboardR) {
                //     for (int i = 0; i < innerArray.rows; i++) {
                //         for (int j = 0; j < innerArray.cols; j++) {
                //             std::cout << innerArray.at<double>(i, j) << ", ";
                //         }
                //     }
                //     printf("\n");
                // }
                // printf("\n");
                // printf("Chessboard Translational Matrix: \n");
                // for (const auto& innerArray : chessboardT) {
                //     for (int i = 0; i < innerArray.rows; i++) {
                //         for (int j = 0; j < innerArray.cols; j++) {
                //             std::cout << innerArray.at<double>(i, j) << ", ";
                //         }
                //     }
                //     printf("\n");
                // }


            }
        }
        //if the user pressed 'h' key, detect the corner points using HarrisCorner Feature.
        else if(key == 'h'){
            //set the harris corner flag to true
            harrisCornerFlag=true;
        }
        if(harrisCornerFlag){
            // detect the corner points and return the frames with detected corner points.
            displayedFrame=harrisCornerFeature(frame);
        }

        if (chessboardDistCoeffs.rows != 0) {

            // extract chess board corners of current frame
            std::vector<cv::Point2f> currCorners;
            bool foundCurrCorners = chessBoardCorners(frame, chessboardPatternSize, currCorners);

            //Display a virtual object in live video if chess board corners are found.
            if (foundCurrCorners) {
                //create empty matrix to store rotational and translation matrix of camera
                cv::Mat rvec, tvec;

                //get the rotational and translation matrix of the camera.
                bool status = solvePnP(chessBoardPoints, currCorners, chessboardCameraMatrix, chessboardDistCoeffs, rvec, tvec);
                
                if (status) {
                    //print out the rotational and translation vector of the camera.
                    // printf("Chess Board rotational vector \n");
                    // for (int i = 0; i < rvec.rows; i++) {
                    //     for (int j = 0; j < rvec.cols; j++) {
                    //         std::cout << rvec.at<double>(i, j) << ", ";
                    //     }
                    //     printf("\n");
                    // }
                    // printf("\n");
                    // printf("Chess Board translational vector \n");
                    // for (int i = 0; i < tvec.rows; i++) {
                    //     for (int j = 0; j < tvec.cols; j++) {
                    //         std::cout << tvec.at<double>(i, j) << ", ";
                    //     }
                    //     printf("\n");
                    // }
                    // printf("\n");

                    // project outside corners on the live video
                    //projectOutsideCorners(displayedFrame, chessBoardPoints, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs);

                    // project a virtual object on the live video
                    projectVirtualObject(displayedFrame, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs);
                }
            }
        }

        //display the live video with detected chessboard corner points, projected outside corners and virtual object on chessboard target
        cv::imshow("Video", displayedFrame);

        //if the user pressed 'q' key, break the loop.
        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    //releases the video and terminates the window created.
    frame.release();
    cv::destroyAllWindows();

    //ends the main function by returning 0 value to main program.
    return 0;
}