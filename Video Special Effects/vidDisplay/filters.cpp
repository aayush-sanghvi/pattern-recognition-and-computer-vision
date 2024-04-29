/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 27st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

Functions for designing different filters and modifying the frames
of videos using the OpenCV library.
*/

//To include all the standard libraries
#include <iostream>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include "faceDetect.h"

//initialise variables to be used later in the program.
int flag=0; 
int i=0;
int j=0;
std::vector<cv::Rect> rectVector;

/*
Task 4: int greyscale(cv::Mat &source, cv::Mat &destination);
Function applies a transformation to create a different greyscale 
representation by subtracting the red pixel value and assigning 
that value to each channel of the output frame.

Function key: h

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where alternative greyscaled 
frames are stored

return: Returns 0 upon successful execution of the function
*/
int greyscale(cv::Mat &source, cv::Mat &destination)
{
    cv::Mat gray = cv::Mat::zeros(source.rows,source.cols,source.type());
    for(int i=0;i<source.rows;i++)
    {
        uchar *o = source.ptr<uchar>(i);
        uchar *g = gray.ptr<uchar>(i);
        for(int j=0;j<source.cols;j++)
        {
            g[j*3]=(o[j*3]-255);
            g[j*3+1]=(o[j*3]-255);
            g[j*3+2]=(o[j*3]-255);
        }
    }   
    destination=gray; 
    return(0);
}

/*
Task 5: int sepia(cv::Mat &source, cv::Mat &destination);
Function calculates sepia toned values based on predefined 
coefficients mentioned and stores the value in each channel of the
output frames.

Function key: e

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sepia-toned frames are 
stored

return: Returns 0 upon successful execution of the function
*/
int sepia( cv::Mat &source, cv::Mat &destination )
{
    cv::Mat sepia = cv::Mat::zeros(source.rows,source.cols,source.type());
    for(int i=0;i<source.rows;i++)
    {
        uchar *o = source.ptr<uchar>(i);
        uchar *s = sepia.ptr<uchar>(i);
        for(int j=0;j<source.cols;j++)
        {
            s[j*3]=fmin((o[j*3]*0.131)+(o[j*3+1]*0.534)+(o[j*3+2]*0.272),255);
            s[j*3+1]=fmin((o[j*3]*0.168)+(o[j*3+1]*0.686)+(o[j*3+2]*0.349),255);
            s[j*3+2]=fmin((o[j*3]*0.189)+(o[j*3+1]*0.769)+(o[j*3+2]*0.393),255);
        }   
    }  
    destination=sepia; 
    return(0);
}

/*
Task 6.A: int blur5x5_1(cv::Mat &source, cv::Mat &destination);
First implementation of 5x5 blur filter
Function calculates a weighted average to blur the pixels of the 
image. It implements convolution directly by using 5x5 filter to 
compute blurred output.

Function key: c

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where blurred frames are stored

return: Returns 0 upon successful execution of the function
*/
int blur5x5_1(cv::Mat &source, cv::Mat &destination)
{
    destination=source;
    for(int i=2;i<source.rows-2;i++)
    {
        for(int j=2;j<source.cols-2;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum=(source.at<cv::Vec3b>(i-2,j-2)[k])+(2*source.at<cv::Vec3b>(i-2,j-1)[k])+(4*source.at<cv::Vec3b>(i-2,j)[k])+(2*source.at<cv::Vec3b>(i-2,j+1)[k])+(source.at<cv::Vec3b>(i-2,j+2)[k])+
                        (2*source.at<cv::Vec3b>(i-1,j-2)[k])+(4*source.at<cv::Vec3b>(i-1,j-1)[k])+(8*source.at<cv::Vec3b>(i-1,j)[k])+(4*source.at<cv::Vec3b>(i-1,j+1)[k])+(2*source.at<cv::Vec3b>(i-1,j+2)[k])+
                        (4*source.at<cv::Vec3b>(i,j-2)[k])+(8*source.at<cv::Vec3b>(i,j-1)[k])+(16*source.at<cv::Vec3b>(i,j)[k])+(8*source.at<cv::Vec3b>(i,j+1)[k])+(4*source.at<cv::Vec3b>(i,j+2)[k])+
                        (2*source.at<cv::Vec3b>(i+1,j-2)[k])+(4*source.at<cv::Vec3b>(i+1,j-1)[k])+(8*source.at<cv::Vec3b>(i+1,j)[k])+(4*source.at<cv::Vec3b>(i+1,j+1)[k])+(2*source.at<cv::Vec3b>(i+1,j+2)[k])+
                        (source.at<cv::Vec3b>(i+2,j-2)[k])+(2*source.at<cv::Vec3b>(i+2,j-1)[k])+(4*source.at<cv::Vec3b>(i+2,j)[k])+(2*source.at<cv::Vec3b>(i+2,j+1)[k])+(source.at<cv::Vec3b>(i+2,j+2)[k]);
                sum/=100;
                destination.at<cv::Vec3b>(i,j)[k]=sum;
            }  
        }   
    }
    return(0);
}

/*
Task 6.B: int blur5x5_2(cv::Mat &source, cv::Mat &destination);
Second implementation of 5x5 blur filter using separable 1x5 filters
Function calculates a weighted average to blur the pixels of the 
image. It implements convolution using separable 1x5 filter to 
compute blurred output.

Function key: b

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where blurred frames are stored

return: Returns 0 upon successful execution of the function
*/
int blur5x5_2(cv::Mat &source, cv::Mat &destination)
{
    //destination=source;
    //cv::Mat blur;
    for(int i=2;i<source.rows-2;i++)
    {
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *dst=destination.ptr<cv::Vec3b>(i);
        for(int j=2;j<source.cols-2;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum=(mid[j-2][k])+(2*mid[j-1][k])+(4*mid[j][k])+(2*mid[j+1][k])+(mid[j+2][k]);
                sum/=10;
                dst[j][k]=sum;
            }  
        } 
        for(int j=2;j<source.cols-2;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum = (destination.at<cv::Vec3b>(i - 2,j)[k]) + (2 * destination.at<cv::Vec3b>(i - 1,j)[k]) +
                          (4 * destination.at<cv::Vec3b>(i,j)[k]) + (2 * destination.at<cv::Vec3b>(i + 1,j)[k]) +
                          (destination.at<cv::Vec3b>(i + 2,j)[k]);
                sum /= 10;
                destination.at<cv::Vec3b>(i,j)[k] = sum;
            }  
        }   
    }
    return(0);
}

/*
Task 7.A: int sobelX3x3(cv::Mat &source, cv::Mat &destination);
Function caluculates the gradient along the X direction using
sobel filters and stores them in output frame. It implements 
convolution using separable 1x3 filter to compute Sobel output.

Function key: x

Note: The commented section in the code also produces the same 
      output but does not implements convultion with separable filter

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sobel filter along 
x-direction is implemented.

return 0: Returns 0 upon successful execution of the function
*/
int sobelX3x3(cv::Mat &source, cv::Mat &destination) {
    cv::Mat X;
    X.create(source.size(), CV_16SC3);
    for (int i = 1; i < source.rows-1; i++)
    {
        cv::Vec3b *up = source.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *down = source.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *x = X.ptr<cv::Vec3s>(i);
        for(int j = 0; j < source.cols; j++)
        {
            for (int k = 0; k < source.channels(); k++){
                x[j][k] = 1 * down[j][k] + 2 * mid[j][k] + 1 * up[j][k]/4;
            }
        }
    }
    destination.create(X.size(), CV_16SC3);
    for (int i = 0; i < X.rows; i++) {
        cv::Vec3s *x = X.ptr<cv::Vec3s>(i);
        cv::Vec3s *dst = destination.ptr<cv::Vec3s>(i);
        for (int j = 1; j < X.cols-1; j++){
            for(int k = 0; k < X.channels(); k++){
                dst[j][k] = -1*x[j-1][k] + 1*x[j+1][k];
            }
        }
    }
    return 0;
}

/*
Task 7.B: int sobelY3x3(cv::Mat &source, cv::Mat &destination);
Function caluculates the gradient along the Y direction using
sobel filters and stores them in output frame. It implements 
convolution using separable 1x3 filter to compute Sobel output.

Function key: y

Note: The commented section in the code also produces the same 
      output but does not implements convultion with separable filter 

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sobel filter along 
y-direction is implemented.

return 0: Returns 0 upon successful execution of the function
*/
int sobelY3x3(cv::Mat &source, cv::Mat &destination)
{
    cv::Mat Y;
    Y.create(source.size(), CV_16SC3);
    for (int i = 1; i < source.rows-1; i++)
    {
        cv::Vec3b *up = source.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *down = source.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *y = Y.ptr<cv::Vec3s>(i);
        for(int j = 0; j < source.cols; j++)
        {
            for (int k = 0; k < source.channels(); k++){
                y[j][k] = 1 * down[j][k] + 0 * mid[j][k] - 1 * up[j][k];
            }
        }
    }
    destination.create(Y.size(), CV_16SC3);
    for (int i = 0; i < Y.rows; i++) {
        cv::Vec3s *x = Y.ptr<cv::Vec3s>(i);
        cv::Vec3s *dst = destination.ptr<cv::Vec3s>(i);
        for (int j = 1; j < Y.cols-1; j++){
            for(int k = 0; k < Y.channels(); k++){
                dst[j][k] = 1*x[j-1][k] + 2*x[j][k] + 1*x[j+1][k]/4;
            }
        }
    }
    return 0;
}

/*
Task 8: int magnitude(cv::Mat &source, cv::Mat &destination);
Function caluculates the gradient along the X and Y direction using
sobel filters then calculates the magnitude and stores them in 
output frame

Function key: m

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where gradient magnitude are 
stored

return 0: Returns 0 upon successful execution of the function
*/
int magnitude(cv::Mat &source, cv::Mat &destination )
{
    cv::Mat sourcex=source;
    cv::Mat destinationx;
    cv::Mat sourcey=source;
    cv::Mat destinationy;
    sobelX3x3(sourcex,destinationx);
    sobelY3x3(sourcey,destinationy);
    destination.create(destinationx.size(), CV_16SC3);
    for (int i=0;i < destinationx.rows; i++){
        cv::Vec3s *x = destinationx.ptr<cv::Vec3s>(i);
        cv::Vec3s *y = destinationy.ptr<cv::Vec3s>(i);
        cv::Vec3s *dst = destination.ptr<cv::Vec3s>(i);
        for(int j=0; j<destinationx.cols; j++){
            for(int k=0; k<destinationx.channels(); k++){
                dst[j][k]=sqrt((x[j][k]*x[j][k])+(y[j][k]*y[j][k]));
            }
        }
    }
    return 0;
}

/*
Task 9: int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels);
Function makes use of blur5x5_2 function to blur the frames of video 
and then quantized pixel values to discrete levels as mentioned.

Function key: l

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where the blurred and quantised
frames are stored.

int levels: Number of levels each pixel needs to be quantized

return: Returns 0 upon successful execution of the function
*/
int blurQuantize(cv::Mat& source, cv::Mat &destination, int levels)
{
    cv::Mat blur;
    blur5x5_2(source,destination);
    double bucketSize = 255.0/levels;
    for (int i=0;i<destination.rows;i++)
    {
        for(int j=0; j<destination.cols;j++)
        {
            for(int k=0; k<destination.channels();k++)
            {
                double xt = destination.at<cv::Vec3b>(i,j)[k]/bucketSize;
                double xf = xt*bucketSize;
                destination.at<cv::Vec3b>(i,j)[k]=xf;
            }
        }
    }
    return(0);
}

/*
Task 11.A: int coloredFaces(cv::Mat& source, cv::Mat &destination);
Make the face colorful, while the rest of the image is greyscale.
Function detects faces in each frame of the video and then converts 
the image outside the faces into grayscale.

Function key: j

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where pixels outside the 
detected faces are grayscale.

return: Returns 0 upon successful execution of the function
*/
int coloredFaces(cv::Mat &source, cv::Mat &destination)
{
    cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    detectFaces(destination,rectVector);
    drawBoxes(destination,rectVector);
    cv::cvtColor(destination,destination,cv::COLOR_GRAY2BGR);
    for (const auto& rect : rectVector)
    {
        for (int i=rect.y;i<rect.y+rect.height;i++)
        {
            for (int j=rect.x;j<rect.x+rect.width;j++)
            {
                destination.at<cv::Vec3b>(i,j)=source.at<cv::Vec3b>(i,j);
            }
        }
    }
    return(0);
}

/*
Task 11.B: int blurBackground(cv::Mat& source, cv::Mat &destination);
Blur the image outside of found faces.
Function detects faces in each frame of the video and then blurs 
the pixels which are not a part of the face.

Function key: v

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where pixels outside the 
detected faces are blurred

return: Returns 0 upon successful execution of the function
*/
int blurBackground(cv::Mat &source, cv::Mat &destination)
{
    cv::Mat src=source;
    cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    detectFaces(destination,rectVector);
    cv::cvtColor(destination,destination,cv::COLOR_GRAY2BGR);
    blur5x5_2(src,destination);
    drawBoxes(destination,rectVector);
    for (const auto& rect : rectVector)
    {
        for (int i=rect.y;i<rect.y+rect.height;i++)
        {
            for (int j=rect.x;j<rect.x+rect.width;j++)
            {
                destination.at<cv::Vec3b>(i,j)=source.at<cv::Vec3b>(i,j);
            }
        }
    }
    return(0);
}

/*
Task 11.C: int haloEffect(cv::Mat& source, cv::Mat &destination);
Put sparkles in a halo above the face.
Function detects faces in each frame of the video and puts a halo 
above every detected face. Additionally, I have place a small tilak 
on head of every face.

Function key: a

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where halo is created above 
every detected face.

return: Returns 0 upon successful execution of the function
*/
int haloEffect(cv::Mat& source, cv::Mat &destination)
{
    cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    detectFaces(destination,rectVector);
    destination=source;
    for (const auto& rect : rectVector)
    {
        cv::ellipse(destination,cv::Point(rect.x+rect.width/2,rect.y+rect.height/3.75),cv::Size(4,10),0,0,360,cv::Scalar(28,96,244),-1);
        int X=rect.x+rect.width/2;
        int Y=rect.y+rect.height/200;
        for (int i=0;i<1600;i++)
        {
            double angle = i*3.14/25;
            int xp=(int)(X+100*cos(angle));
            int yp=(int)(Y+15*sin(angle));
            cv::circle(destination,cv::Point(xp,yp-40),5,cv::Scalar(106,211,255),-1);
        }
    }
    return(0);
}

/*
Extension: int extensions(cv::Mat &source, cv::Mat &destination);
Function first generates a sepia toned frames and then creates a 
vignetting effect to the image. 

Function key: z

Note: The commented section in the code implements the blur5x5_2
      function as designed for Task 6.B which provides the same 
      output as the below code. But due to excessive computational 
      time, in-built GuassianBlur function of OpenCV has been 
      implemented below. Difference between output of both the 
      functions have been addressed in the Report.

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sepia toned frames with 
a vignetting effect are stored

return: Returns 0 upon successful execution of the function
*/
int extensions(cv::Mat &source, cv::Mat &destination)
{
    
    cv::Mat sepiaf = cv::Mat::zeros(source.rows,source.cols,source.type());
    cv::Mat mask = cv::Mat(source.size(), CV_8U);
    cv::Point center(source.cols/2, source.rows/2);
    int radius = std::min(center.x, center.y);
    cv::ellipse(mask, center, cv::Size(radius*1, radius*1), 0, 0, 360, cv::Scalar(255), -1);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    // for(int i=0;i<40;i++)
    // {
    //     blur5x5_2(mask, mask);
    // }
    cv::GaussianBlur(mask,mask,cv::Size(0,0),60);
    sepia(source,sepiaf); 
    cv::multiply(sepiaf, mask, sepiaf,1.0/250.0);
    destination=sepiaf; 
    return(0);
}

/*
Extension: int cartoonization(cv::Mat &source, cv::Mat &destination,int n);
cv::Mat edge_mark(cv::Mat img,int line_size, int blur_val);
Function performs blurring and quantization on the images and edge masking 
and creates a cartoon image of the given image using two functions 
edge_mark, blurQuantize. The edge_mark is another custom method using 
multiple opencv function like cvtcolor, medianblur, adaptivethreshold.

Function key: p


Arguments: (cartoonization)
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sepia toned frames with 
a vignetting effect are stored

int n: this is the level of quantization used in the blurQuantize function

return: Returns 0 upon successful execution of the function
------

Arguments: (edge_mark)
cv::Mat img: Input frames of original video upon which filter 
needs to be implemented

int line_size: Interger value for the edge masking line size

int blur_val: this is the value of blur used by the opencv function 
medianblur

return: returns a cv:Mat edges which has the values of the edges 
that are formed.
------
*/
cv::Mat edge_mark(cv::Mat img,int line_size, int blur_val){
    cv::Mat gray,gray_blur,edges;
    cvtColor(img,gray,cv::COLOR_RGB2GRAY);
    medianBlur(gray,gray_blur,blur_val);
    adaptiveThreshold(gray_blur, edges, 255, cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY, line_size, blur_val);
    return edges;
}
int cartoonization(cv::Mat &source, cv::Mat &destination,int n)
{
    cv::Mat edge_val,blurred;
    edge_val = edge_mark(source,19,9);
    blurQuantize(source,destination,n);
    cv::bilateralFilter(destination,blurred,9,200,200);
    cv::bitwise_and(blurred,blurred,destination,edge_val);
    return 0;
}

/*
int filter(int Flag,cv::Mat &source, cv::Mat &destination)
This function calls different filter function based on the flag 
value which will be set by imageModification function.

Arguments:
int Flag: Flag number based on which which filter function needs to be 
called 

cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sepia tone of frames 
stored

return: Returns 0 upon successful execution of the function
*/
int filter(int Flag,cv::Mat &source, cv::Mat &destination)
{
    //Task 3: Display greyscale live video
    if(Flag==1)
    {
        cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    }
    //Displays original video if the value of flag is 2
    else if(Flag==2)
    {
        destination=source;
    }
    //Task 4: Display alternative greyscale live video
    else if(Flag==3)
    {
        greyscale(source,destination);   
    }
    //Task 5: Implement a Sepia tone filter
    else if(Flag==4)
    {
        sepia(source,destination);    
    }
    //Task 6.B: Implement a 5x5 blur filter
    else if(Flag==5)
    {
        blur5x5_2(source,destination);
        blur5x5_2(destination,source);
        blur5x5_2(source,destination);   
    }
    //Task 6.A: Implement a 5x5 blur filter
    else if(Flag==6)
    {
        blur5x5_1(source,destination);
        blur5x5_1(destination,source);
        blur5x5_1(source,destination);   
    }
    //Task 7.A: Implement a 3x3 Sobel X filter as separable 1x3 filters
    else if(Flag==7)
    {
        cv::Mat dst;
        sobelX3x3(source,dst);
        cv::convertScaleAbs(dst,destination);   
    }
    //Task 7.B: Implement a 3x3 Sobel Y filter as separable 1x3 filters
    else if(Flag==8)
    {
        cv::Mat dst;
        sobelY3x3(source,dst);  
        cv::convertScaleAbs(dst,destination);   
    }
    //Task 8: Implement a function that generates a gradient magnitude image from the X and Y Sobel images
    else if(Flag==9)
    {
        cv::Mat dst;
        magnitude(source,dst);   
        cv::convertScaleAbs(dst,destination);  
    }
    //Task 9: Implement a function that blurs and quantizes a color image
    else if(Flag==10)
    {
        blurQuantize(source,destination,255);   
    }
    //Task 10: Detect faces in an image
    else if(Flag==11)
    {
        cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
        detectFaces(destination,rectVector);
        drawBoxes(destination,rectVector);
    }
    //Task 11.A: Implement three more effects on your video (Make the face colorful, while the rest of the image is greyscale.)
    else if(Flag==12)
    {
        coloredFaces(source,destination);
    }
    ////Task 11.B: Implement three more effects on your video (Blur the image outside of found faces.)
    else if(Flag==13)
    {
        blurBackground(source,destination);
    }
    //Task 11.C: Implement three more effects on your video (Put sparkles in a halo above the face.)
    else if(Flag==14)
    {
        haloEffect(source,destination);
    }
    //Extension: Vignetting effect to sepia tone frame.
    else if(Flag==15)
    {
        extensions(source,destination);
    }
    //Extension: Cartoonization of the image using filters and edge masks
    else if(Flag==16)
    {
        cartoonization(source,destination,2);
    }
    //Display the original video if no key pressed
    else
    {
        destination=source;
    }
    
    return 0;
}

/*
int imageModification(cv::Mat &source,cv::Mat &destination)
This function monitors keyboard input for key-press events. 
Depending on the pressed key, it assigns a specific value to a 
flag. Once the flag is set, the program calls the filter function,
applying a filter to the original input video frames.

It also allows user to save frame of video in build directory by 
pressing "s" key.

Arguments:
cv::Mat &source: Input frames of original video upon which filter 
needs to be implemented

cv::Mat &destination: Output frames where sepia tone of frames 
stored

return: Returns 0 upon successful execution of the function
*/
int imageModification(cv::Mat &source,cv::Mat &destination)
{   
    char key = cv::waitKey(1);

    switch (key){
        case 's':
            while(i<50){
            std::string j=std::to_string(i);
            cv::imwrite("frame"+j+".jpeg",source);
            i+=1;
            break;
            }
        case 'g': flag = 1;break;
        case 'o': flag = 2;break;
        case 'h': flag = 3;break;
        case 'e': flag = 4;break;
        case 'b': flag = 5;break;
        case 'c': flag = 6;break;
        case 'x': flag = 7;break;
        case 'y': flag = 8;break;
        case 'm': flag = 9;break;
        case 'l': flag = 10;break;
        case 'f': flag = 11;break;
        case 'j': flag = 12;break;
        case 'v': flag = 13;break;
        case 'a': flag = 14;break;
        case 'z': flag = 15;break;
        case 'p': flag = 16;break;
        default: break;
    }
    filter(flag,source,destination);
    return(0);
}