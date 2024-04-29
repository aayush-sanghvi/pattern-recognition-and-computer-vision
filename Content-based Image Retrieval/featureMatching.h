/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 11th Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To calculate different feature vectors and distance metrics and provide
the best matches of target image using the feature vectors and distance
metrics based on user input.
*/

#include <vector>

#ifndef FEATUREMATCHING_H
#define FEATUREMATCHING_H

/*
int featureMatch(char targetImage[],char feature[], int number, int N)
This function takes in an user input for the type of distance metrics 
the user would like to choose and the compute the distance metrics and
provide the top N result for the target image provided.

Arguments:
char targetImage[]: Holds the target image. 

char feature[]: Holds the csv file which contains the feature vector 
                of all image in database.

int N: Number of best matchs to be displayed.

return: Returns 0 upon successful execution of the function
*/
int featureMatch(char targetImage[],char feature[], int number, int N);

/*
gradientMagnitude(cv::Mat &source, cv::Mat &magnitude, cv::Mat &orientation)
Function caluculates the gradient along the X and Y direction using
sobel filters then calculates the magnitude and orientation and stores 
them in output frame

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &magnitude: Output matrix where gradient magnitude are 
stored

cv::Mat &orientation: Output matrix where orientation are stored

return: Returns 0 upon successful execution of the function
*/
int gradientMagnitude(cv::Mat &source, cv::Mat &magnitude, cv::Mat &orientation);

#endif