/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 26th Jan 2024
CS5330- Pattern Recognition and Computer Vision.

Implement various stages to detect an object in live video 
and recognise the object with help of different classifying 
algorithms.
*/

// Include all necessary libraries.
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>


#ifndef IMAGEPROCESSING
#define IMAGEPROCESSING

/*
cv::Mat feature(cv::Mat &image, cv::Mat &threshold, cv::Mat &clean, cv::Mat &seg, cv::Mat &dst, cv::Moments &m, std::vector<int> &largeLabels, cv::Mat &centroids)
Function extracts features from a segmented image, including the bounding box and orientation 
of each large connected component. It first executes thresholding and cleanup functions to the 
input image, followed by segmentation function to identify large connected components. Then, for each 
large component, it computes its moments, orientation, and bounding box. It then draws an arrow 
and bounding box on the original image representing orientation and region boundary respectively.

Arguments:
cv::Mat &image: Holds the input binary image to be segmented.

cv::Mat &threshold: Holds the outputed thresholded image.

cv::Mat &clean: Holds the cleaned up image.

cv::Mat &seg: Holds the segmented image.

cv::Mat &dst: Holds the image that contains top 3 largest regions.

cv::Moments &m: Stores the moments of each regions.

std::vector<int> &largeLabels: Hold the lables of top 3 largest region.

return image: Returns image with arrow and rectangle detecting each region in the frame.
*/
std::vector<std::vector<float>> feature(cv::Mat &image, cv::Mat &threshold, cv::Mat &clean, cv::Mat &seg, cv::Mat &region, cv::Moments &m, std::vector<int> &largeLabels, cv::Mat &centroids);

/*
int calculateHuMoments(cv::Moments &m, std::vector<double> &huMoments)
Function calculates the seven Hu moments from the given moments and stores them in a vector. 

Arguments:
cv::Moments &m: Holds the moments of each regions.

std::vector<int> &huMoments: Stores the 7 Hu Moments of each regions.

return 0: Returns 0 upon successful execution of the function.
*/
int calculateHuMoments(cv::Moments &m, std::vector<double> &huMoments);

/*
std::string getClassName(char c) 
Function maps a character representing a class label to its corresponding class name using a 
predefined map. The map associates each character with a string representing a class name.  

Arguments:
char c: Holds the character represnting the object name.

return myMap[c]: Returns a map of each identified object.
*/
char* getClassName(char c);

/*
std::string classifier(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature)
Function compares a given feature vector against a database of known feature vectors. It 
iterates through each known feature vector, computes the Euclidean distance between the current 
feature vector and each known feature vector, and selects the closest match within a specified 
threshold.

Arguments:
std::vector<std::vector<double>> featureVectors: Contains the feature vector of known objects(recognised objects).

std::vector<std::string> classNames: Contains the object names of known objects.

std::vector<double> currentFeature: Contains the feature vector of unknown objects(to be recognised objects).

return className: Returns object name corresponding to the given unknow feature vector.
*/
std::string classifier(std::vector<std::vector<float>> featureVectors, std::vector<char*> classNames, std::vector<float> currentFeature);

std::vector<std::string> kNN(std::vector<std::vector<float>> featureVectors, std::vector<char*> classNames, std::vector<float> currentFeature,int k);

#endif