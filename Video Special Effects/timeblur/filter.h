/*
Aayush H Sanghvi
Spring 2024 semseter
Date:- 27st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

Header file for the filter.cpp file which includes all
functions to be executed by the main program.
*/

#include <opencv4/opencv2/opencv.hpp>

#ifndef FILTER
#define FILTER

//defines functions to be called in the main program.
int image_modification(cv::Mat &source,cv::Mat &destination);
int blur5x5_1(cv::Mat &source, cv::Mat &destination);
int blur5x5_2(cv::Mat &source, cv::Mat &destination);
int greyscale(cv::Mat &source, cv::Mat &destination);

#endif
