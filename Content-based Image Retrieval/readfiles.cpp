/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 11th Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To calculate different feature vectors and distance metrics and provide
the best matches of target image using the feature vectors and distance
metrics based on user input.
*/

// include all necessary libraries.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "csv_util.h"
#include "featureMatching.h"

/*
int baselineImage(cv::Mat image, char buffer[256])
Function computes the baseline feature vector, it creates a 7x7 matrix 
at the center of the image and takes the RGB value of each pixel in the 
7x7 matrix as a feature for all the images in the database.

Arguments:
cv::Mat image: Holds each image present in tbhe database in matrix form.

char buffer[256]: Hold the path name for each image in database

return 0: Returns 0 upon successful execution of the function
*/
int baselineImage(cv::Mat image, char buffer[256]){
    std::vector<float> baselineVector;
    for(int i=(image.rows/2)-3;i<=(image.rows/2)+3;i++){
        cv::Vec3b *img = image.ptr<cv::Vec3b>(i);
        for(int j=(image.cols/2)-3;j<=(image.cols/2)+3;j++){
            baselineVector.push_back(img[j][0]);
            baselineVector.push_back(img[j][1]);
            baselineVector.push_back(img[j][2]);
        }
    }
    append_image_data_csv("baseline_feature.csv",buffer,baselineVector);
    return(0);
}

/*
int textureImage(cv::Mat image, char buffer[256])
Function computes the texture vector, it calculates the magnitude and orientation of
each image in data base and creates a 2-D matrix which holds the histogram of magnitude
vs orientation.
This histogram in then normalized.

Arguments:
cv::Mat image: Holds each image present in tbhe database in matrix form.

char buffer[256]: Hold the path name for each image in database

return 0: Returns 0 upon successful execution of the function
*/
int textureImage(cv::Mat image, char buffer[256]){
    std::vector<float> textureVector;
    cv::Mat magnitude;
    cv::Mat orientation;
    gradientMagnitude(image,magnitude,orientation);
    //printf("%i   %i\n",magnitude.size(),orientation.size());
    cv::Mat text = cv::Mat::zeros(cv::Size(8,8),CV_32FC1);
    for(int i=0;i<=magnitude.rows;i++){
        uchar *m = magnitude.ptr<uchar>(i);
        uchar *o = orientation.ptr<uchar>(i);
        for(int j=0;j<=magnitude.cols;j++){
            float mag= m[j]/255; float ori= o[j]/255;
            int mindex=(int)((mag*7)+0.5);int oindex=(int)((ori*7)+0.5);
            text.at<float>(mindex,oindex)++;
        }
    }
    text/=(magnitude.rows*magnitude.cols);
    for(int i=0;i<text.rows;i++){
        for(int j=0;j<text.cols;j++){
            textureVector.push_back(text.at<float>(i,j));
        }
    }
    append_image_data_csv("texture_feature.csv",buffer,textureVector);
    return(0);
}

/*
int histogramImage(cv::Mat image, char buffer[256])
Function computes a 3-D RGB chromaticity histogram feature vector, it takes 
the RGB value of each pixel in the image, calculates the r-index and g-index
and b-index and creates a 3-D histogram 8x8x8 matrix giving us 512 bins.
This histogram was then normalized.

Arguments:
cv::Mat image: Holds each image present in tbhe database in matrix form.

char buffer[256]: Hold the path name for each image in database

return 0: Returns 0 upon successful execution of the function
*/
int histogram3DImage(cv::Mat image, char buffer[256]){
    std::vector<float> histogram3DVector;
    int dim[]={8,8,8};
    cv::Mat hist3dtop = cv::Mat::zeros(3,dim,CV_32FC1);
    for(int i=0;i<=image.rows/2;i++){
        cv::Vec3b *im = image.ptr<cv::Vec3b>(i);
        for(int j=0;j<=image.cols;j++){
            float B = im[j][0]; float G = im[j][1]; float R = im[j][2];
            float div = (B+G+R)>0.0? (B+G+R) : 1.0;
            float r = R/div; float g = G/div; float b =B/div;
            int rindex = (int)(r*(7)+0.5); int gindex = (int)(g*(7)+0.5); int bindex = (int)(b*(7)+0.5);
            hist3dtop.at<float>(rindex,gindex,bindex)+=1;
        }
    }
    cv::Mat hist3dbottom = cv::Mat::zeros(3,dim,CV_32FC1);
    for(int i=(image.rows/2)+1;i<=image.rows;i++){
        cv::Vec3b *im = image.ptr<cv::Vec3b>(i);
        for(int j=0;j<=image.cols;j++){
            float B = im[j][0]; float G = im[j][1]; float R = im[j][2];
            float div = (B+G+R)>0.0? (B+G+R) : 1.0;
            float r = R/div; float g = G/div; float b =B/div;
            int rindex = (int)(r*(7)+0.5); int gindex = (int)(g*(7)+0.5); int bindex = (int)(b*(7)+0.5);
            hist3dbottom.at<float>(rindex,gindex,bindex)+=1;
        }
    }
    hist3dtop/=(image.rows*image.cols);
    hist3dbottom/=(image.rows*image.cols);
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            for(int k=0; k< 8; k++){
                histogram3DVector.push_back(hist3dtop.at<float>(i,j,k));
            }
        }
    }
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            for(int k=0; k<8; k++){
                histogram3DVector.push_back(hist3dbottom.at<float>(i,j,k));
            }
        }
    }
    append_image_data_csv("histogram3D_feature.csv",buffer,histogram3DVector);
    return(0);
}

/*
int histogramImage(cv::Mat image, char buffer[256])
Function computes a 2-D RG chromaticity histogram feature vector, it takes 
the RGB value of each pixel in the image, calculates the  r-index and g-index
and creates a 2-D histogram of 16x16 matrix giving us 256 bins.
This histogram was then normalized

Arguments:
cv::Mat image: Holds each image present in tbhe database in matrix form.

char buffer[256]: Hold the path name for each image in database

return 0: Returns 0 upon successful execution of the function
*/
int histogramImage(cv::Mat image, char buffer[256]){
    std::vector<float> histogramVector;
    cv::Mat hist = cv::Mat::zeros(cv::Size(16,16),CV_32FC1);
    for(int i=0;i<=image.rows;i++){
        cv::Vec3b *im = image.ptr<cv::Vec3b>(i);
        for(int j=0;j<=image.cols;j++){
            float B = im[j][0]; float G = im[j][1]; float R = im[j][2];
            float div = (B+G+R)>0.0? (B+G+R) : 1.0;
            float r = R/div; float g = G/div;
            int rindex = (int)(r*(15)+0.5); int gindex = (int)(g*(15)+0.5);
            hist.at<float>(rindex,gindex)++;
        }
    }
    hist/=(image.rows*image.cols);
    for(int i=0;i<hist.rows;i++){
        for(int j=0;j<hist.cols;j++){
            histogramVector.push_back(hist.at<float>(i,j));
        }
    }
    append_image_data_csv("histogram_feature.csv",buffer,histogramVector);
    return(0);
}

int main(int argc, char *argv[]) {

    // removes the csv file in present in the directory.
    std::remove("baseline_feature.csv");
    std::remove("histogram_feature.csv");
    std::remove("histogram3D_feature.csv");
    std::remove("texture_feature.csv");

    // declare all the necessary variables.
    int N;
    DIR *dirp;
    struct dirent *dp;
    char dirname[256];
    char buffer[256];
    char feature[256];
    char targetImage[256];
    int number;

    // check for sufficient arguments
    if( argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // get the directory path, target image, integer values to display those many best match.
    strcpy(dirname, argv[1]);
    strcpy(targetImage, argv[2]);
    N=atol(argv[3]);
    printf("Processing directory %s\n", dirname );

    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {
        
        // check if the file is an image
        if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            //reads an image and stores it in opencv Mat datatype variable.
            cv::Mat image = cv::imread(buffer);

            // checks if the image read is a valid image (not empty).
            if(image.empty()){
                printf("Not a valid image. Please read a valid image");
                exit(-1);
            }

            // execute the functions to calculate the all feature vector in the database. 
            baselineImage(image,buffer);
            histogramImage(image,buffer);
            histogram3DImage(image,buffer);
            textureImage(image,buffer);  
        }
    }
    
    //create a command line GUI to ask for type of feature vector the user would like to view output for.
    printf("\n");
    printf("Please enter the number for what kind of feature vector you want to generate.\n");
    printf("\n");
    printf("1: 7x7 matrix in the center of the image.\n");
    printf("2: 2-D RG chromaticity histogram of the image.\n");
    printf("3: 3-D RGB hustogram of the image.\n");
    printf("4: Texture features of the image.\n");
    printf("5: Deep network embeddings of the image.\n"); 

    // save the feature vector in a variable according to user input.
    std::cin>>number;
    if(number==1){
        strcpy(feature,"/home/aayush/PRCV/Project-2/build/baseline_feature.csv");
    }
    else if(number==2){
        strcpy(feature,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv");
    }
    else if(number==3){
        strcpy(feature,"/home/aayush/PRCV/Project-2/build/histogram3D_feature.csv");
    }
    else if(number==4){
        strcpy(feature,"/home/aayush/PRCV/Project-2/build/texture_feature.csv");
    }
    else if(number==5){
        strcpy(feature,"/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv");
    }
    else{
        printf("Please enter a correct number");
    }

    // execute the feature matching function to get the top N bext results.
    featureMatch(targetImage,feature,number,N);
    return(0);
}