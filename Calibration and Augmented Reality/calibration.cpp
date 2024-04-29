/*
Aayush H Sanghvi and Yogi H Shah
Spring 2024 semseter
Date:- 18th Mar 2024
CS5330- Pattern Recognition and Computer Vision.

To calibrate the camera and then use the calibration 
to generate virtual objects in a scene.
*/

//Include all the standard and necessary libraries.
#include <iostream>
#include "opencv4/opencv2/opencv.hpp"

/*
std::vector<cv::Vec3f> worldCoordinates(cv::Size patternSize);
Function generates a set of 3D world coordinates representing a grid pattern with the 
specified size.

Arguments:
cv::Size patternSize: Pattern size for which 3-D co-ordinates needs to be generated
frames are stored

return points: 3-D world co-ordinates
*/
std::vector<cv::Vec3f> worldCoordinates(cv::Size patternSize) {
    std::vector<cv::Vec3f> points;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            cv::Vec3f coordinates = cv::Vec3f(j, -i, 0);
            points.push_back(coordinates);
        }
    }
    return points;
}

/*
bool chessBoardCorners(cv::Mat &frame, cv::Size patternSize, std::vector<cv::Point2f> &corners);
Function extracts corners of chessboard pattern from a given input image.

Arguments:
cv::Mat &frame: Input image from which chessboard corners needs to be extracted.

cv::Size patternSize: Pattern size for which chessboard corners needs to be extracted.

std::vector<cv::Point2f> &corners: vector to store extracted chessboard corners.

return foundCorners: true value if corner extraction was successful else false
*/
bool chessBoardCorners(cv::Mat &frame, cv::Size patternSize, std::vector<cv::Point2f> &corners) {
    bool findCorners = findChessboardCorners(frame, patternSize, corners);
    //printf("%i\n",foundCorners);
    if (findCorners) {
        cv::Mat grayscale;
        cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        cv::Size subPixWinSize(10, 10);
        cv::TermCriteria termCrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 1, 0.1);
        cornerSubPix(grayscale, corners, subPixWinSize, cv::Size(-1, -1), termCrit);
    }
    return findCorners;
}

/*
std::vector<cv::Vec3f> objectPoints();
Function generates set of 3D points which are the vertices of pyramid

return objectPoints: 3-D points of a virtual pyramid
*/
std::vector<cv::Vec3f> objectPoints() {
    std::vector<cv::Vec3f> objectPoints;
    objectPoints.push_back(cv::Vec3f(2, -1, 1));
    objectPoints.push_back(cv::Vec3f(2, -5, 1));
    objectPoints.push_back(cv::Vec3f(6, -1, 1));
    objectPoints.push_back(cv::Vec3f(6, -5, 1));
    objectPoints.push_back(cv::Vec3f(4, -3, 4));
    return objectPoints;
}

/*
void virtualObjects(cv::Mat &frame, std::vector<cv::Point2f> p);
Function creates a virtual pyramid on the given input image

Arguments:
cv::Mat &frame: Input image on which virtual object needs to be drawn

std::vector<cv::Point2f> p: vector of vertices of pyramid
*/
void virtualObjects(cv::Mat &frame, std::vector<cv::Point2f> p) {
    cv::line(frame, p[0], p[1], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[0], p[2], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[1], p[3], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[2], p[3], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[0], p[4], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[1], p[4], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[2], p[4], cv::Scalar(147, 100, 160), 2);
    cv::line(frame, p[3], p[4], cv::Scalar(147, 100, 160), 2);
    cv::circle(frame,p[4],5,cv::Scalar(147, 20, 255), 2);
}

/*
void projectOutsideCorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rmat, cv::Mat tmat, cv::Mat cameraMatrix, cv::Mat distCoeffs);
Function projects a set of 3D world coordinates onto an input image using a specified rotation 
vector, translation vector, camera matrix, and distortion coefficients. It then draws circles 
at specific points from the projected image points onto the input frame

Arguments:
cv::Mat &frame: Input image on which circle needs to be drawn on outside points

std::vector<cv::Vec3f> points: chess board corner points

cv::Mat rvec: rotational matrix

cv::Mat tvec: translational matrix

cv::Mat cameraMatrix: camera matrix 

cv::Mat distCoeffs: distance coefficient
*/
void projectOutsideCorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rmat, cv::Mat tmat, cv::Mat cameraMatrix, cv::Mat distCoeffs) {
    std::vector<cv::Point2f> imagePoints;
    projectPoints(points, rmat, tmat, cameraMatrix, distCoeffs, imagePoints);
    int index[] = {0, 8, 45, 53};
    for (int i : index) {
        circle(frame, imagePoints[i], 5, cv::Scalar(147, 20, 255), 4);
    }
}

/*
void projectVirtualObject(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat cameraMatrix, cv::Mat distCoeffs);
Function projects virtual object onto an input image using a specified rotation vector, 
translation vector, camera matrix, and distortion coefficients. It then draws a virtual 
objects on the projected points

Arguments:
cv::Mat &frame: Input image on which circle needs to be drawn on outside points

cv::Mat rvec: rotational matrix

cv::Mat tvec: translational matrix

cv::Mat cameraMatrix: camera matrix 

cv::Mat distCoeffs: distance coefficient
*/
void projectVirtualObject(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat cameraMatrix, cv::Mat distCoeffs) {
    std::vector<cv::Vec3f> objectPoint = objectPoints();
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(objectPoint, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    for (int i = 0; i < projectedPoints.size(); i++) {
        circle(frame, projectedPoints[i], 1, cv::Scalar(147, 20, 255), 4);
    }
    virtualObjects(frame, projectedPoints);
}

/*
cv::Mat harrisCornerFeature(cv::Mat &frame);
Function uses HarrisCorner Feature detection to detect the corner points in the given image.

Arguments:
cv::Mat &frame: Input image for which corner points needs to be detected

return frame: returns the frames which contains circle on each detected corner points
*/
cv::Mat harrisCornerFeature(cv::Mat &frame){
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
    cv::Mat dst = cv::Mat::zeros(grayscale.size(), CV_32FC1);
    cornerHarris(grayscale, dst, 2, 3, 0.04);
    double min, max;
    cv::minMaxLoc(dst, &min, &max);
    float thresh = 0.1 * max;
    for (int i = 0; i < dst.rows ; i++) {
        for(int j = 0; j < dst.cols; j++) {
            if (dst.at<float>(i,j) > thresh) {
                cv::circle(frame, cv::Point(j,i), 1, cv::Scalar(147, 20, 255), 2);
            }
        }
    }
    return frame;
}