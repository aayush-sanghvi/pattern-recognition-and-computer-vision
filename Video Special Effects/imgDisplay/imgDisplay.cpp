/*
Aayush H Sanghvi & Yogi Hetal Shah
Spring 2024 semseter
Date:- 27st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To display image and apply different filters 
using the OpenCV library on that image.
*/

//Includes all the necessay libraries required.
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>


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
    cv::Mat X = cv::Mat::zeros(source.rows,source.cols,CV_16SC3);
    cv::Mat Y = cv::Mat::zeros(source.rows,source.cols,CV_16SC3);
    cv::Mat M = cv::Mat::zeros(source.rows,source.cols/2,CV_16SC3);
    for(int i=1;i<source.rows-1;i++)
    {
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *x=X.ptr<cv::Vec3b>(i);
        cv::Vec3b *y=Y.ptr<cv::Vec3b>(i);
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sumx=(-1*mid[j-1][k])+(mid[j+1][k]);
                int sumy = -source.at<cv::Vec3b>(i - 1,j)[k] +
                          source.at<cv::Vec3b>(i + 1,j)[k];
                x[j][k]=sumx/8;
                Y.at<cv::Vec3b>(i,j)[k] = sumy/8;
            }  
        } 
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sumx = X.at<cv::Vec3b>(i - 1,j)[k] +
                          (2 * X.at<cv::Vec3b>(i,j)[k])+ 
                          X.at<cv::Vec3b>(i + 1,j)[k];
                int sumy=(1*y[j-1][k])+(2*y[j+1][k])+(y[j+1][k]);
                y[j][k]=sumy/8;
                X.at<cv::Vec3b>(i,j)[k] = sumx/8;
            }  
        }
    }
    for(int i=1;i<source.rows-1;i++)
    {
        uchar *m=M.ptr<uchar>(i);
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                m[j*3+k]=fmin(255,sqrt((Y.at<cv::Vec3b>(i,j)[k]*Y.at<cv::Vec3b>(i,j)[k]+(X.at<cv::Vec3b>(i,j)[k]*X.at<cv::Vec3b>(i,j)[k]))));
            }
        }
    }
    destination=M;
    return(0); 
}

//Main function
int main(int argc, char *argv[])
{
    //checks if image argument is mentioned.
    if(argc<2)
    {
        printf("Please specify the image to display");
        exit(-1);
    }

    //reads an image and stores it in opencv Mat datatype variable.
    cv::Mat img = cv::imread((std::string)argv[1]);
    cv::Mat dst;

    // checks if the image read is a valid image (not empty).
    if(img.empty())
    {
        printf("Not a valid image. Please read a valid image");
        exit(-1);
    }

    //declare empty images for different functionalities of size of the image read earlier.
    cv::Mat blue = cv::Mat::zeros(img.rows,img.cols,img.type());
    cv::Mat green = cv::Mat::zeros(img.rows,img.cols,img.type());
    cv::Mat red = cv::Mat::zeros(img.rows,img.cols,img.type());
    cv::Mat gray = cv::Mat::zeros(img.rows,img.cols,CV_8U); 

    //Display the size,channels and number of bytes used by each pixel of the image.
    printf("size of the image is %i columns by %i rows \n",img.cols,img.rows);
    printf("Number of channel used by the image is %i \n",img.channels());
    printf("Number of bytes used by each channel is %i \n",(int)(img.elemSize()/img.channels()));

    //creates a named window and displays the image in window using imshow function.
    cv::namedWindow(argv[1],1);
    cv::imshow(argv[1],img);

    //separating the red, green and blue component of the image
    for(int i=0;i<img.rows;i++)
    {
        //define pointers for each channel in image
        uchar *o = img.ptr<uchar>(i);
        uchar *b = blue.ptr<uchar>(i);
        uchar *g = green.ptr<uchar>(i);
        uchar *r = red.ptr<uchar>(i);
        uchar *y = gray.ptr<uchar>(i);
        //iterate through the each pixel and store blue, green and red values in respective variables.
        for(int j=0;j<img.cols;j++)
        {
            b[j*3]=o[j*3];
            b[j*3+1]=0;
            b[j*3+2]=0;
            g[j*3]=0;
            g[j*3+1]=o[j*3+1];
            g[j*3+2]=0;
            r[j*3]=0;
            r[j*3+1]=0;
            r[j*3+2]=o[j*3+2];
            y[j]=(o[j*3+2]*0.114)+(o[j*3+1]*0.587)+(o[j*3]*0.299);//used NTSC formula to convert BGR into grayscale
        }   
    }

    // Calculate the gradient magnitude of an image and display the gradient magnitude of the image
    magnitude(img, dst);
    cv::resize(dst, dst, cv::Size(600,400),cv::INTER_LINEAR);
    cv::namedWindow("texture",cv::WINDOW_AUTOSIZE);
    cv::imshow("texture",dst);

    //checks for key press by the user to either to exit the program or display different colors of the image.
    while(1)
    {
        //if 'q' key is pressed, the program exits
        if(cv::waitKey(0)=='q')
        {
           break;
        }
        //if 'b' key is pressed, blue channel of the image is displayed
        else if(cv::waitKey(0)=='b')
        {
            cv::namedWindow("Blue",cv::WINDOW_AUTOSIZE);
            cv::imshow("Blue",blue);
        }
        //if 'g' key is pressed, blue channel of the image is displayed
        else if(cv::waitKey(0)=='g')
        {
            cv::namedWindow("Green",cv::WINDOW_AUTOSIZE);
            cv::imshow("Green",green);
        }
        //if 'r' key is pressed, blue channel of the image is displayed
        else if(cv::waitKey(0)=='r')
        {
            cv::namedWindow("Red",cv::WINDOW_AUTOSIZE);
            cv::imshow("Red",red);
        }
        //if 'y' key is pressed, grayscale version of the image is displayed
        else if(cv::waitKey(0)=='y')
        {
            cv::namedWindow("Gray",cv::WINDOW_AUTOSIZE);
            cv::imshow("Gray",gray);
        }

    }

    //ends the main function by returning 0 value to main program.
    return 0;
}