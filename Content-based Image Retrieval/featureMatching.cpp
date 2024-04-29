/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 11th Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To calculate different feature vectors and distance metrics and provide
the best matches of target image using the feature vectors and distance
metrics based on user input.
*/

//include all necessary libraries.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "csv_util.h"

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

return 0: Returns 0 upon successful execution of the function
*/
int gradientMagnitude(cv::Mat &source, cv::Mat &magnitude, cv::Mat &orientation)
{
    cv::Mat X = cv::Mat::zeros(source.rows,source.cols,CV_8UC1);
    cv::Mat Y = cv::Mat::zeros(source.rows,source.cols,CV_8UC1);
    cv::Mat M = cv::Mat::zeros(source.rows,source.cols,CV_8UC1);
    cv::Mat O = cv::Mat::zeros(source.rows,source.cols,CV_8UC1);
    cv::cvtColor(source,source,cv::COLOR_BGR2GRAY);
    for(int i=1;i<source.rows-1;i++)
    {
        uchar *x=X.ptr<uchar>(i);
        uchar *y=Y.ptr<uchar>(i);
        for(int j=1;j<source.cols-1;j++)
        {
            int sumx=-source.at<uchar>(i,j-1) + source.at<uchar>(i,j+1);
            int sumy = -source.at<uchar>(i - 1,j) +
                        source.at<uchar>(i + 1,j);
            x[j]=sumx/8;
            Y.at<uchar>(i,j) = sumy/16;
             
        }
        for(int j=1;j<source.cols-1;j++)
        {
            int sumx = X.at<uchar>(i - 1,j) +
                       (2 * X.at<uchar>(i,j))+ 
                       X.at<uchar>(i + 1,j);
            int sumy=(1*y[j-1])+(2*y[j+1])+(y[j+1]);
            y[j]=sumy/8;
            X.at<uchar>(i,j) = sumx/16;  
        }
    }
    for(int i=1;i<source.rows-1;i++)
    {
        uchar *m=M.ptr<uchar>(i);
        uchar *o=O.ptr<uchar>(i);

        for(int j=1;j<source.cols-1;j++)
        {
            m[j]=fmin(255,sqrt((Y.at<uchar>(i,j)*Y.at<uchar>(i,j)+(X.at<uchar>(i,j)*X.at<uchar>(i,j)))));
            o[j]=atan2(Y.at<uchar>(i,j),X.at<uchar>(i,j));
            
        }
    }
    magnitude=M;
    orientation=O;
    return(0); 
}

/*
std::vector<float> TargetImageFeature(char baseline_targetImage[], std::vector<char *> filenames,std::vector<std::vector<float>> data)
Function compares the target image with the images in the data base and stores the 
feature vector of the target image in a variable. 

Argumemts:
char baseline_targetImage[]: Hold the target image.

std::vector<char *> filenames: Holds the path for all the image present in database.

std::vector<std::vector<float>> data: Holds the feature vectors of all the image present in database.

return: Returns a vector type which contains feature vector of the target image.
*/
std::vector<float> TargetImageFeature(char baseline_targetImage[], std::vector<char *> filenames,std::vector<std::vector<float>> data){
    std::vector<float> vectorT;
    for(int i=0;i<filenames.size();i++){
        if(!strcmp(filenames[i],baseline_targetImage)){
            vectorT=data[i];
            break;
        }
    }
    return(vectorT);
}

/*
int histogram3DTargetImageFeature(char histogram3D_targetImage[], std::vector<char *> filenames,std::vector<std::vector<float>> data, std::vector<float> &vectorT, std::vector<float> &vectorB)
Function compares the target image with the images in the data base for 3-D histogram vector
and stores the feature vector of the target image in a variable. 

Argumemts:
char histogram3D_targetImage[]: Hold the target image.

std::vector<char *> filenames: Holds the path for all the image present in database.

std::vector<std::vector<float>> data: Holds the 3-D histogram feature vectors of all 
                                      the image present in database.

return: Returns a vector type which contains feature vector of the target image.
*/
int histogram3DTargetImageFeature(char histogram3D_targetImage[], std::vector<char *> filenames,std::vector<std::vector<float>> data, std::vector<float> &vectorT, std::vector<float> &vectorB){
    for(int i=0;i<filenames.size();i++){
        if(!strcmp(filenames[i],histogram3D_targetImage)){
            for(int j=0;j<data[i].size()/2;j++){
                vectorT.push_back(data[i][j]);
            }
            for(int j=data[i].size()/2;j<=data[i].size();j++){
                vectorB.push_back(data[i][j]);
            }
        }
    }
    return(0);
}

/*
int euclideanDistance(char targetImage[], const char csv_file[], int N)
Function calculates the Euclidean distance metrics of the feature vector for target 
image and images in the database and displays the top N matches for the given target 
images.

Arguments:
char targetImage[]: Hold the target image.

const char csv_file[]: Hold the csv file that contains feature vector of all the image in database.

int N: Number of best matches to display.
*/
int euclideanDistance(char targetImage[], const char csv_file[], int N){
    int count = 0;
    std::map< int, std::string> baseline;
    std::vector<char *> baseline_filenames;
    std::vector<std::vector<float>> baseline_data;
    read_image_data_csv(csv_file,baseline_filenames,baseline_data,0);
    if(strlen(baseline_filenames[0])<13){
        for(int i=0;i<baseline_filenames.size();i++){
            char* str = new char[strlen("/home/aayush/PRCV/Project-2/olympus") + strlen("/") + strlen(baseline_filenames[i]) + 1];
            strcpy(str,"/home/aayush/PRCV/Project-2/olympus");
            strcat(str, "/");
            strcat(str, baseline_filenames[i]);
            baseline_filenames[i]=str;
        }
    }
    std::vector<float> vectorT=TargetImageFeature(targetImage, baseline_filenames,baseline_data);
    std::vector<float> ssd;
    for(const auto& value : baseline_data){
        std::vector<int> vectorI;
        double sum = 0.0;
        for(int i=0;i<baseline_data[i].size();i++){
            vectorI.push_back(value[i]);
        }
        if (vectorT.size() != vectorI.size()) {
            std::cerr << "Error: Baseline vectors must have the same size." << std::endl;
            int a=0;
        }

        for (int i = 0; i < vectorT.size(); ++i) {
            double diff = vectorT[i] - vectorI[i];
            sum += diff * diff;
        }
        ssd.push_back(sqrt(sum));
    }

    for(int i=0;i<baseline_filenames.size();i++){
        baseline[ssd[i]]=baseline_filenames[i];
    }
    printf("Top %i matches for Baseline matching are\n",N);
    for (auto it = baseline.begin(); it != baseline.end() && count < N; it++) {
        if(it==baseline.begin()){
            continue;
        }
        std::cout << it->first << ": " << it->second << std::endl;
        count+=1;
    }
    printf("\n");

    return(0);
}

/*
int histogramIntersection(char histogram_targetImage[], const char csv_file[], int N)
Function calculates the Histogram Intersection distance metrics of the feature vector 
for target image and images in the database and displays the top N matches for the given 
target images.

Arguments:
char histogram_targetImage[]: Hold the target image.

const char csv_file[]: Hold the csv file that contains feature vector of all the image in database.

int N: Number of best matches to display.
*/
int histogramIntersection(char histogram_targetImage[], const char csv_file[], int N){
    int count=0;
    std::vector<float> vectorT;
    std::vector<float> hi;
    std::vector<char *> histogram_filenames; 
    std::vector<std::vector<float>> histogram_data;
    std::map< float, std::string, std::greater<float>> hist;

    read_image_data_csv(csv_file,histogram_filenames,histogram_data,0);
    vectorT=TargetImageFeature(histogram_targetImage,histogram_filenames,histogram_data);
    
    for(const auto& value : histogram_data){
        std::vector<float> vectorI;
        double sum = 0.0;

        for(int i=0;i<value.size();i++){
            vectorI.push_back(value[i]);
        }

        if (vectorT.size() != vectorI.size()) {
            std::cerr << "Error: Histogram vectors must have the same size." << std::endl;
        }

        for (int i=0; i<vectorT.size(); i++){
            sum+=(vectorT[i]>vectorI[i]?vectorI[i]:vectorT[i]);
        }
        hi.push_back(sum);
    }

    for(int i=0;i<histogram_filenames.size();i++){
        hist[hi[i]]=histogram_filenames[i];
    }
    printf("Top %i matches for Histogram matching are\n",N);
    for (auto it = hist.begin(); it != hist.end() && count < N; it++) {
        if(it==hist.begin()){
            continue;
        }
        std::cout << it->first << ": " << it->second << std::endl;
        count+=1;
    }
    printf("\n");
    return(0);
}

/*
int histogramIntersection(char histogram_targetImage[], const char csv_file[], int N)
Function calculates the Histogram Intersection distance metrics of the feature vector 
for top half and bottom half of the target image and images in the database separately,
computes the weighted average and displays the top N matches for the given target images.

Arguments:
char histogram_targetImage[]: Hold the target image.

const char csv_file[]: Hold the csv file that contains feature vector of all the image in database.

int N: Number of best matches to display.
*/
int histogram3DIntersection(char histogram_targetImage[], const char csv_file[],int N){
    int count=0;
    std::vector<float> vectorT;
    std::vector<float> vectorB;
    std::vector<float> hi3d;
    std::vector<char *> histogram3D_filenames;
    std::vector<std::vector<float>> histogram3D_data;
    std::map< float, std::string, std::greater<float>> hist3D;
    read_image_data_csv(csv_file,histogram3D_filenames,histogram3D_data,0);

    histogram3DTargetImageFeature(histogram_targetImage,histogram3D_filenames,histogram3D_data,vectorT,vectorB);

    for(const auto& value : histogram3D_data){
        std::vector<float> vectorIT;
        std::vector<float> vectorIB;
        double sum = 0.0;

        for(int i=0;i<value.size()/2;i++){
            vectorIT.push_back(value[i]);
        }
        for(int i=value.size()/2;i<=value.size();i++){
            vectorIB.push_back(value[i]);
        }

        if ((vectorT.size() != vectorIT.size()) && (vectorB.size() != vectorIB.size())) {
            std::cerr << "Error: 3D Histogram vectors must have the same size." << std::endl;
        }

        for (int i=0; i<vectorT.size(); i++){
            sum+=(vectorT[i]>vectorIT[i]?vectorIT[i]:vectorT[i]);
            //printf("target image=%f    database image=%f     sum=%f\n",vectorT[i],vectorI[i],sum);
        }
        for (int i=0; i<vectorB.size(); i++){
            sum+=(vectorB[i]>vectorIB[i]?vectorIB[i]:vectorB[i]);
            //printf("target image=%f    database image=%f     sum=%f\n",vectorT[i],vectorI[i],sum);
        }
        hi3d.push_back(sum);
        //printf("%f\n",sum);
    }

    for(int i=0;i<histogram3D_filenames.size();i++){
        hist3D[hi3d[i]]=histogram3D_filenames[i];
    }
    printf("Top %i matches for Multi-histogram matching are\n",N);
    for (auto it = hist3D.begin(); it != hist3D.end() && count < N; it++) {
        if(it==hist3D.begin()){
            continue;
        }
        std::cout /*<< it->first << ": " */<< it->second << std::endl;
        count+=1;
    }
    printf("\n");
    return(0);

}

/*
int histogramIntersection(char histogram_targetImage[], const char csv_file[], int N)
Function calculates the texture distance metrics and histogram distance metrics using 
texture feature vector and histogram feature vector of the target image and images in 
data base, computes the weighted average and displays the top N matches for the given 
target images.

Arguments:
char texture_targetImage[]: Hold the target image.

const char csv_file[]: Hold the csv file that contains feature vector of all the image in database.

int N: Number of best matches to display.
*/
int textureMetric(char texture_targetImage[],const char csv_file[], int N){
    std::vector<float> vectorTT;
    std::vector<float> vectorTC;
    std::vector<float> vectorIT;
    std::vector<float> vectorIC;
    double textureDistance=0.0;
    double colorDistance=0.0;
    int count=0;
    std::map< float, std::string, std::greater<float>> texture;
    std::vector<float> tm;

    std::vector<char *> texture_filenames;
    std::vector<std::vector<float>> texture_data;
    read_image_data_csv(csv_file,texture_filenames,texture_data,0);
    
    std::vector<char *> histogram3D_filenames;
    std::vector<std::vector<float>> histogram3D_data;
    read_image_data_csv("/home/aayush/PRCV/Project-2/build/histogram3D_feature.csv",histogram3D_filenames,histogram3D_data,0);

    
    vectorTT=TargetImageFeature(texture_targetImage,texture_filenames,texture_data);
    vectorTC=TargetImageFeature(texture_targetImage,histogram3D_filenames,histogram3D_data);

    for(const auto& value : texture_data){
        for(int i=0;i<128;i++){
            vectorIT.push_back(value[i]);
        }
    }
    for(const auto& value : histogram3D_data){
        for(int i=0;i<value.size();i++){
            vectorIC.push_back(value[i]);
        }
    }
    for(const auto& value : texture_data){
        double sum=0.0;
        for(int i=0; i<value.size(); i++){
            textureDistance += vectorTT[i]-vectorIT[i];
        }
        for(int i=0; i<value.size(); i++){
            colorDistance += vectorTC[i]-vectorIC[i];
        }
        sum=(textureDistance+colorDistance)/2.0;
        tm.push_back(sum);
    }

    for(int i=0;i<texture_filenames.size();i++){
        texture[tm[i]]=texture_filenames[i];
    }
    printf("Top %i matches for Texture-histogram matching are\n",N);
    for (auto it = texture.begin(); it != texture.end() && count < N; it++) {
        std::cout /*<< it->first << ": " */<< it->second << std::endl;  
        count+=1;
    }
    printf("\n");
    return(0);
}

/*
int custom(char custom_targetImage[],const char csv_file_1[],const char csv_file_2[],int N)
Function calculates the custom distance metrics using deep network vector and histogram feature
vector of the target image and images in data base, computes the weighted average and displays 
the top N matches for the given target images.

Arguments:
char custom_targetImage[]: Hold the target image.

const char csv_file_1[]: Hold the csv file that contains feature vector of all the image in database.

const char csv_file_2[]: Hold the csv file of histogram feature vector of all the image in database.

int N: Number of best matches to display.
*/
int custom(char custom_targetImage[],const char csv_file_1[],const char csv_file_2[],int N){
    std::vector<float> kela_1;
    std::vector<float> kela_2;
    std::vector<float> kela;
    std::map< float, std::string, std::greater<float>> bananamap;
    int count=0;
    std::vector<char *> histogram_filenames;
    std::vector<std::vector<float>> histogram_data;
    read_image_data_csv(csv_file_1,histogram_filenames,histogram_data,0);

    std::vector<char *> deepnetwork_filenames;
    std::vector<std::vector<float>> deepnetwork_data;
    read_image_data_csv(csv_file_2,deepnetwork_filenames,deepnetwork_data,0);
    if(strlen(deepnetwork_filenames[0])<13){
        for(int i=0;i<deepnetwork_filenames.size();i++){
            char* str = new char[strlen("/home/aayush/PRCV/Project-2/olympus") + strlen("/") + strlen(deepnetwork_filenames[i]) + 1];
            strcpy(str,"/home/aayush/PRCV/Project-2/olympus");
            strcat(str, "/");
            strcat(str, deepnetwork_filenames[i]);
            deepnetwork_filenames[i]=str;
        }
    }

    std::vector<float> vectorHistogram=TargetImageFeature(custom_targetImage,histogram_filenames,histogram_data);
    std::vector<float> vectorDeepNetwork=TargetImageFeature(custom_targetImage,deepnetwork_filenames,deepnetwork_data);

    for(const auto& value : histogram_data){
        std::vector<float> vectorI;
        double sum = 0.0;

        for(int i=0;i<value.size();i++){
            vectorI.push_back(value[i]);
        }

        if (vectorHistogram.size() != vectorI.size()) {
            std::cerr << "Error: Histogram vectors must have the same size." << std::endl;
        }

        for (int i=0; i<vectorHistogram.size(); i++){
            sum+=(vectorHistogram[i]>vectorI[i]?vectorI[i]:vectorHistogram[i]);
        }
        kela_1.push_back(sum);
    }

    for(const auto& value : deepnetwork_data){
        std::vector<int> vectorI;
        double sum = 0.0;
        for(int i=0;i<deepnetwork_data[i].size();i++){
            vectorI.push_back(value[i]);
        }
        if (vectorDeepNetwork.size() != vectorI.size()) {
            std::cerr << "Error: Baseline vectors must have the same size." << std::endl;
            int a=0;
        }

        for (int i = 0; i < vectorDeepNetwork.size(); ++i) {
            double diff = vectorDeepNetwork[i] - vectorI[i];
            sum += diff * diff;
        }
        kela_2.push_back(sqrt(sum));
    }
    
    if (kela_1.size() != kela_2.size()) {
        std::cerr << "Error: vectors must have the same size." << std::endl;
    }

    for(int i=0;i<deepnetwork_filenames.size();i++){
        for(int j=0;j<histogram_filenames.size();j++){
            if(!strcmp(deepnetwork_filenames[i],histogram_filenames[j])){
                kela.push_back(kela_1[j]*11-kela_2[i]/2);

            }
        }
        
    }
    for(int i=0;i<deepnetwork_filenames.size();i++){
        bananamap[kela[i]]=deepnetwork_filenames[i];
    }
    printf("Top %i matches for Custom designed distance metrics are\n",N);
    for (auto it = bananamap.begin(); it != bananamap.end() && count<N; it++) {
        if(it==bananamap.begin()){
            continue;
        }
        std::cout << it->first << ": " << it->second << std::endl;
        count+=1;
    }
    printf("\n");
    return(0);
}

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
int featureMatch(char targetImage[],char feature[],int number,int N) {
    int value;
    if(number==1){
        printf("Which distance metrics do you want to choose?\n");
        printf("1: Euclidean Distance\n");
        printf("2: Histogram Intersection\n");
        printf("3: Split Histogram Intersection\n");
        printf("4: Texture Distance Metrics\n");
        printf("5: Custom designed Distance Metrics\n");
        std::cin>>value;
        if(value==1){
            euclideanDistance(targetImage,feature,N);
        }
        else if(value==2){
            histogramIntersection(targetImage,feature,N);
        }
        else if(value==3){
            histogram3DIntersection(targetImage,feature,N);
        }
        else if(value==4){
            textureMetric(targetImage,feature,N);
        }
        else if(value==5){
            custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);
        }
    }
    else if(number==2){
        printf("Euclidean Distance cannot be calculated for this feature type.\n");
        printf("Which distance metrics do you want to choose?\n");
        printf("1: Histogram Intersection\n");
        printf("2: Split Histogram Intersection\n");
        printf("3: Texture Distance Metrics\n");
        printf("4: Custom designed Distance Metrics\n");
        std::cin>>value;
        if(value==1){
            histogramIntersection(targetImage,feature,N);
        }
        else if(value==2){
            histogram3DIntersection(targetImage,feature,N);
        }
        else if(value==3){
            textureMetric(targetImage,feature,N);
        }
        else if(value==4){
            custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);
        }
    }
    else if(number==3){
        printf("Euclidean Distance cannot be calculated for this feature type.\n");
        printf("Which distance metrics do you want to choose?\n");
        printf("1: Histogram Intersection\n");
        printf("2: Split Histogram Intersection\n");
        printf("3: Texture Distance Metrics\n");
        printf("4: Custom designed Distance Metrics\n");
        std::cin>>value;
        if(value==1){
            histogramIntersection(targetImage,feature,N);
        }
        else if(value==2){
            histogram3DIntersection(targetImage,feature,N);
        }
        else if(value==3){
            textureMetric(targetImage,feature,N);
        }
        else if(value==4){
            custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);
        }
    }
    else if(number==4){
        printf("Euclidean Distance cannot be calculated for this feature type.\n");
        printf("Which distance metrics do you want to choose?\n");
        printf("1: Histogram Intersection\n");
        printf("2: Split Histogram Intersection\n");
        printf("3: Texture Distance Metrics\n");
        printf("4: Custom designed Distance Metrics\n");
        std::cin>>value;
        if(value==1){
            histogramIntersection(targetImage,feature,N);
        }
        else if(value==2){
            histogram3DIntersection(targetImage,feature,N);
        }
        else if(value==3){
            textureMetric(targetImage,feature,N);
        }
        else if(value==4){
            custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);
        }
    }
    else if(number==5){
        printf("For this feature, either Euclidean Distance metrics can be calculated or Custom designed distance metric can be calculated.\n");
        printf("Which distance metrics do you want to choose?\n");
        printf("1: Euclidean Distance\n");
        printf("2: Custom designed Distance Metrics\n");
        std::cin>>value;
        if(value==1){
            euclideanDistance(targetImage, feature, N);
        }
        else if(value==2){
            custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);
        }        
    }
    // euclideanDistance(targetImage, feature, N);
    // histogramIntersection(targetImage, feature, N);
    // histogram3DIntersection(targetImage,feature, N);
    // textureMetric(targetImage, feature, N);
    // custom(targetImage,"/home/aayush/PRCV/Project-2/build/histogram_feature.csv","/home/aayush/PRCV/Project-2/build/ResNet18_olym.csv",N);

    return(0);
}