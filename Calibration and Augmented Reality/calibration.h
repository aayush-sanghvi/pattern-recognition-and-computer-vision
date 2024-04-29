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
#include <opencv2/opencv.hpp>

#ifndef CALIBRATION
#define CALIBRATION

/*
std::vector<cv::Vec3f> worldCoordinates(cv::Size patternSize);
Function generates a set of 3D world coordinates representing a grid pattern with the 
specified size.

Arguments:
cv::Size patternSize: Pattern size for which 3-D co-ordinates needs to be generated
frames are stored

return points: 3-D world co-ordinates
*/
std::vector<cv::Vec3f> worldCoordinates(cv::Size patternSize);

/*
bool chessBoardCorners(cv::Mat &frame, cv::Size patternSize, std::vector<cv::Point2f> &corners);
Function extracts corners of chessboard pattern from a given input image.

Arguments:
cv::Mat &frame: Input image from which chessboard corners needs to be extracted.

cv::Size patternSize: Pattern size for which chessboard corners needs to be extracted.

std::vector<cv::Point2f> &corners: vector to store extracted chessboard corners.

return foundCorners: true value if corner extraction was successful else false
*/
bool chessBoardCorners(cv::Mat &frame, cv::Size patternSize, std::vector<cv::Point2f> &corners);

/*
void projectOutsideCorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rvec, cv::Mat tvec, cv::Mat cameraMatrix, cv::Mat distCoeffs);
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
void projectOutsideCorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rvec, cv::Mat tvec, cv::Mat cameraMatrix, cv::Mat distCoeffs);

/*
void projectVirtualObject(cv::Mat &frame, cv::Mat rmat, cv::Mat tmat, cv::Mat cameraMatrix, cv::Mat distCoeffs);
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
void projectVirtualObject(cv::Mat &frame, cv::Mat rmat, cv::Mat tmat, cv::Mat cameraMatrix, cv::Mat distCoeffs);

/*
cv::Mat harrisCornerFeature(cv::Mat &frame);
Function uses HarrisCorner Feature detection to detect the corner points in the given image.

Arguments:
cv::Mat &frame: Input image for which corner points needs to be detected

return frame: returns the frames which contains circle on each detected corner points
*/
cv::Mat harrisCornerFeature(cv::Mat &frame);

#endif //PROJ4_PROCESSORS_H