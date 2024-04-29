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
#include <limits>
#include "csv_util.h"
/*
cv::Mat thresholding(cv::Mat image)
Function performs thresholding on an input image using a specified threshold value. 
It converts the input image to grayscale and iterates through each pixel, setting 
its value to either 255 or 0 based on whether it is below or above the threshold value, 
respectively

Arguments:
cv::Mat image: Holds current frame of live video in matrix form.
return dst: Returns thresholded image.
*/
cv::Mat thresholding(cv::Mat image){
    const int Threshold=90;
    cv::Mat dst;
    cv::cvtColor(image,dst,cv::COLOR_BGR2GRAY);
    for(int i=0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            if(dst.at<uchar>(i,j)<=Threshold){
                dst.at<uchar>(i,j)=255;
            }
            else{
                dst.at<uchar>(i,j)=0;
            }
        }
    }
    return(dst);
}

/*
cv::Mat grassfire(const cv::Mat &image)
Function implements the grassfire algorithm to compute the Euclidean distance transform 
of a binary input image. It iterates through each pixel of the image, assigning a distance 
value based on the nearest foreground pixel.

Arguments:
cv::Mat image: Holds current frame of live video in matrix form.
return dst: Returns distance-transformed image.
*/
cv::Mat grassfire(const cv::Mat &image) {

  cv::Mat dst(image.rows, image.cols, CV_32S);
  int up, left;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      if (image.at<uchar>(i, j) == 255) {
        up = i == 0 ? 0 : dst.at<int>(i - 1, j);
        left = j == 0 ? 0 : dst.at<int>(i, j - 1);
        dst.at<int>(i, j) = 1 + std::min(up, left);
      } else {
        dst.at<int>(i, j) = 0;
      }
    }
  }

  int down, right;
  for (int i = image.rows - 1; i >= 0; i--) {
    for (int j = image.cols - 1; j >= 0; j--) {
      if (image.at<uchar>(i, j) == 255) {
        down = i == image.rows - 1 ? 0 : dst.at<int>(i + 1, j);
        right = j == image.cols - 1 ? 0 : dst.at<int>(i, j + 1);
        dst.at<int>(i, j) = std::min(dst.at<int>(i, j), 1 + std::min(down, right));
      } else {
        dst.at<int>(i, j) = 0;
      }
    }
  }
  return(dst);
}

/*
int dilate_custom(cv::Mat &src,cv::Mat &dst)
Function checks the 8 connected neighbours and creates a new binary image.
if any of the neighbours have a foreground color then the current pixel 
is given as Foreground value.

Arguments:
cv::Mat src: Holds binary image which needs to be dilated 
return dst: Returns dilated image.
*/
int dilate_custom(cv::Mat &src,cv::Mat &dst){
    cv::Mat dimg = cv::Mat::zeros(src.size(),CV_8UC1);
    for (int i=1; i<src.rows;i++){
        for (int j=1;j<src.cols;j++){
            if(src.at<uchar>(i-1,j)==255 || src.at<uchar>(i+1,j)==255 || src.at<uchar>(i,j-1)==255 || src.at<uchar>(i,j+1)==255 || src.at<uchar>(i-1,j-1)==255 || src.at<uchar>(i+1,j-1)==255 || src.at<uchar>(i+1,j-1)==255 || src.at<uchar>(i-1,j+1)==255){
               dimg.at<uchar>(i,j)= 255;
            }
        }
    }
    dst = dimg;
    //imshow("erode", dst);
    return 0;
}

/*
int erode_custom(cv::Mat &src,cv::Mat &dst)
Function checks the 4 neighbours and creates a new binary image.
if all of the neighbours have a foregrounf color then the current 
pixel is given as Foreground value.

Arguments:
cv::Mat src: Holds binary image which needs to be eroroded 
return dst: Returns eroded image.
*/
int erode_custom(cv::Mat &src,cv::Mat &dst){
    cv::Mat eimg = cv::Mat::zeros(src.size(),CV_8UC1);
    for (int i=1; i<src.rows;i++){
        for (int j=1;j<src.cols;j++){
            if(src.at<uchar>(i-1,j)==255 && src.at<uchar>(i+1,j)==255 && src.at<uchar>(i,j-1)==255 && src.at<uchar>(i,j+1)==255 ){
                eimg.at<uchar>(i,j)= 255;
            }
        }
    }
    dst = eimg;
    //imshow("dilate", dst);
    return 0;
}



/*
cv::Mat cleanup(cv::Mat &image)
Function aims to clean up a binary image by first performing dilation to fill in all the noise
inside the regions and then erroding thrice and then dilating twice to regain the orignal size 
of the image.

Arguments:
cv::Mat image: Holds current frame of live video in matrix form.
return cleanedImage: Returns cleaned up image.
*/
cv::Mat cleanup(cv::Mat &image){
    cv::Mat cleanedImage;
    cv::Mat erodedImage(image.rows,image.cols,CV_8UC1);

    dilate_custom(image,erodedImage);
    erode_custom(erodedImage,erodedImage);
    erode_custom(erodedImage,erodedImage);
    erode_custom(erodedImage,erodedImage);
    dilate_custom(erodedImage,erodedImage);
    dilate_custom(erodedImage,cleanedImage);
    return(cleanedImage);
}

/*
cv::Mat segmentation(cv::Mat &image,cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids, std::vector<int> &largeLabels)
Function performs image segmentation using the connected components algorithm. It takes an input
binary image and computes connected components, storing various statistics about each component. 
It then identifies large connected components based on a specified threshold and assigns unique 
colors to them.

Arguments:
cv::Mat &image: Holds the input binary image to be segmented.
cv::Mat &dst: Stores the output image containing the labeled connected components.
cv::Mat &stats: Stores the statistics about each connected component.
cv::Mat &centroids: Hold the centroids of each connected component.
std::vector<int> &largeLabels: Vector to store labels of large connected components.
return segmentedImage: Returns a segmented image with unique colors assigned to each component.
*/
cv::Mat segmentation(cv::Mat &image, cv::Mat &dst, cv::Mat &stats, cv::Mat &centroids, std::vector<int> &largeLabels){
    int nlabels=cv::connectedComponentsWithStats(image, dst, stats, centroids, 8);
    largeLabels.clear();

    cv::Mat regions = cv::Mat::zeros(1, nlabels-1, CV_32S);
    cv::Mat largestregions;
    for (int i = 1; i < nlabels; i++) {
        int region = stats.at<int>(i, cv::CC_STAT_AREA);
        regions.at<int>(i-1) = region;
    }
    if (regions.cols > 0) {
        cv::sortIdx(regions, largestregions, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
    }

    std::vector<cv::Vec3b> colors(nlabels, cv::Vec3b(0, 0, 0));
    
    int N=3;
    N = (N < largestregions.cols) ? N : largestregions.cols;
    int THRESHOLD = 3000;
    for (int i = 0; i < N; i++) {
        int label = largestregions.at<int>(i) + 1;
        if (stats.at<int>(label, cv::CC_STAT_AREA) > THRESHOLD) {
            colors[label] = cv::Vec3b(20*(i+1),40*(i+1),80*(i+1));
            largeLabels.push_back(label); 
        }
    }
    cv::Mat segmentedImage = cv::Mat::zeros(dst.size(), CV_8UC3);
    for(int i = 0; i < segmentedImage.rows; i++) {
        for (int j = 0; j < segmentedImage.cols; j++) {
            int label = dst.at<int>(i, j);
            segmentedImage.at<cv::Vec3b>(i, j) = colors[label];
        }
    }
    return(segmentedImage);
}

/*
int calculateHuMoments(cv::Moments &m, std::vector<double> &huMoments)
Function calculates the seven Hu moments from the given moments and stores them in a vector. 

Arguments:
cv::Moments &m: Holds the moments of each regions.
std::vector<int> &huMoments: Stores the 7 Hu Moments of each regions.
return 0: Returns 0 upon successful execution of the function.
*/
int calculateHuMoments(cv::Moments &m, std::vector<double> &huMoments){
    double hu[7];
    cv::HuMoments(m, hu);
    for (double d : hu) {
        huMoments.push_back(d);
    }
    return(0);
}
/*
std::string getClassName(char c) 
Function maps a character representing a class label to its corresponding class name using a 
predefined map. The map associates each character with a string representing a class name.  

Arguments:
char c: Holds the character represnting the object name.
return myMap[c]: Returns a map of each identified object.
*/
char* getClassName(char c) {
    std::map<char, char*> myMap {
            {'p', "pen"}, {'a', "alligator"}, {'h', "hammer"}, {'g', "glasses"},
            {'r', "round"}, {'c', "credit card"}, {'b', "bottle"}, {'k', "key"},
            {'m', "mouse"}, {'x', "binder clip"},{'d',"dice"},
            {'w', "wallet"}, {'s', "credit card"}, {'y', "pliers"}
    };
    return myMap[c];
}


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
std::vector<std::vector<float>> feature(cv::Mat &image, cv::Mat &threshold, cv::Mat &clean, cv::Mat &seg, cv::Mat &dst, cv::Moments &m, std::vector<int> &largeLabels, cv::Mat &centroids){
    cv::Mat stats,largeRegions;
    cv::Mat region;
    std::vector<std::vector<float>> final_vector;
    
    threshold=thresholding(image);
    clean=cleanup(threshold);
    seg=segmentation(clean, dst, stats, centroids, largeLabels);
    std::cout<<"\n";
    for (int n = 0; n < largeLabels.size(); n++) {
        int label = largeLabels[n];
        std::vector<float> vectors;
        region = (dst == label);
        m = moments(region, true);
        double alpha = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);
        int maxX = std::numeric_limits<int>::min(), minX = std::numeric_limits<int>::max(), maxY = std::numeric_limits<int>::min(), minY = std::numeric_limits<int>::max();
        for (int i = 0; i < region.rows; i++) {
            for (int j = 0; j < region.cols; j++) {
                if (region.at<uchar>(i, j) == 255) {
                    int projectedX = (i -  centroids.at<double>(label, 0)) * cos(alpha) + (j -  centroids.at<double>(label, 1)) * sin(alpha);
                    int projectedY = -(i - centroids.at<double>(label, 0)) * sin(alpha) + (j -  centroids.at<double>(label, 1)) * cos(alpha);
                    maxX = std::max(maxX, projectedX);
                    minX = std::min(minX, projectedX);
                    maxY = std::max(maxY, projectedY);
                    minY = std::min(minY, projectedY);
                }
            }
        }
        
        int lengthX = maxX - minX;
        int lengthY = maxY - minY;
        
        cv::Point centroid = cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        cv::Size size = cv::Size(lengthX, lengthY);
        alpha= alpha * 180.0 / CV_PI;
        
        cv::RotatedRect boundingBox(centroid, size, alpha * 180.0 / CV_PI);

        double length = 100.0;
        double edge1 = length * sin(alpha);
        double edge2 = sqrt(length * length - edge1 * edge1);
        double xPrime = centroids.at<double>(label, 0) + edge2; 
        double yPrime = centroids.at<double>(label, 1) + edge1;
        cv::arrowedLine(image, cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), cv::Point(xPrime, yPrime), cv::Scalar(255,0,0), 3);

        cv::Rect brect = boundingBox.boundingRect();
        rectangle(image, brect, cv::Scalar(255,0,0));

        // percentage fill
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        float percent_fill = (area*100)/(lengthX*lengthY);
        std::cout<<"region "<<label<<":"<<percent_fill<<", ";
        vectors.push_back(percent_fill);

        // Height-width ratio
        float ratio = lengthY / lengthX;
        vectors.push_back(ratio);

        // Hu moments calculations
        std::vector<double> huMoments;
        calculateHuMoments(m, huMoments);
        vectors.insert(vectors.end(), huMoments.begin(), huMoments.end());

        final_vector.push_back(vectors);
    }
    return(final_vector);
}



/*
double euclideanDistance(std::vector<double> features1, std::vector<double> features2)
Function calculates the Euclidean distance metrics of the feature vector for the given two vectors

Arguments:
std::vector<double> features1: Holds the feature values of known object.
std::vector<double> features1: Holds the feature values of unknown object.
return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2)): Returns euclidean distance of the given feature vectors.
*/
float euclideanDistance(std::vector<float> features1, std::vector<float> features2) {
    float sum1 = 0, sum2 = 0, sumDifference;
    for (int i = 0; i < features1.size(); i++) {
        sumDifference += (features1[i] - features2[i]) * (features1[i] - features2[i]);
        sum1 += features1[i] * features1[i];
        sum2 += features2[i] * features2[i];
    }
    return(sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2)));
}

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
std::string classifier(std::vector<std::vector<float>> featureVectors, std::vector<char*> classNames, std::vector<float> currentFeature) {
    float THRESHOLD = 0.15;
    float distance = DBL_MAX;
    char* className = " ";
    for (int i = 0; i < featureVectors.size(); i++) { // loop the known features to get the closed one
        std::vector<float> dbFeature = featureVectors[i];
        
        char* dbClassName = classNames[i];
        float curDistance = euclideanDistance(dbFeature, currentFeature);
        if (curDistance < distance && curDistance < THRESHOLD) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

/*
std::vector<std::string> kNN(std::vector<std::vector<double>> featureVectors, std::vector<std::string> classNames, std::vector<double> currentFeature, int k)
Function compares a given feature vector against a database of known feature vectors. It 
iterates through each known feature vector, computes the Euclidean distance between the current 
feature vector and each known feature vector, and complies a list of distances with these vectors
and then return the top k feature distance with name.

Arguments:
std::vector<std::vector<double>> featureVectors: Contains the feature vector of known objects(recognised objects).
std::vector<std::string> classNames: Contains the object names of known objects.
std::vector<double> currentFeature: Contains the feature vector of unknown objects(to be recognised objects).
int k: Contains the number of nearest k neighbours.`
return className Vector: Returns object name corresponding to the given unknow feature vector.
*/
std::vector<std::string> kNN(std::vector<std::vector<float>> featureVectors, std::vector<char*> classNames, std::vector<float> currentFeature, int k) {
    // Calculate distances
    std::vector<std::pair<float, int>> distances; // Pair of distance and index
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        float distance = euclideanDistance(featureVectors[i], currentFeature);
        distances.push_back(std::make_pair(distance, i));
    }
    // Sort distances
    std::sort(distances.begin(), distances.end());
    
    // Runs extract the first k neighbours and stores in a string vector
    std::vector<std::string> nearestLabels;
    for (const auto& pair:distances) {
        if (pair.second <k){
        nearestLabels.push_back(classNames[pair.second]);
        std::cout<<classNames[pair.second]<<" : "<< pair.first<<std::endl;
        }
    }
    return nearestLabels;
}


