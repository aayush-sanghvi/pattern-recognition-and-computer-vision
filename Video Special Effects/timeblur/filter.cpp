/*
Aayush H Sanghvi
Spring 2024 semseter
Date:- 27st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To design different filters and modify the frames of 
videos using the OpenCV library.
*/

//To include all the standard libraries
#include <iostream>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include "faceDetect.h"

//initialise variables to be used later in the program.
int flag=0; 
int i=0;
std::vector<cv::Rect> rectVector;


//second implementation of 5x5 blur filter using separable 1x5 filters
int blur5x5_2(cv::Mat &source, cv::Mat &destination)
{

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
    return 0;
}

int extraEffects3(cv::Mat& source, cv::Mat &destination)
{
    cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    detectFaces(destination,rectVector);
    cv::cvtColor(destination,destination,cv::COLOR_GRAY2BGR);
    destination=source;
    for (const auto& rect : rectVector)
    {
        cv::ellipse(destination,cv::Point(rect.x+rect.width/2,rect.y+rect.height/3.75),cv::Size(4,10),0,0,360,cv::Scalar(28,96,244),-1);
        int X=rect.x+rect.width/2;
        int Y=rect.y+rect.height/200;
        for (int i=0;i<100;i++)
        {
            double angle = i*3.14/25;
            int xp=(int)(X+65*cos(angle));
            int yp=(int)(Y+15*sin(angle));
            cv::circle(destination,cv::Point(xp,yp-25),5,cv::Scalar(106,211,255),-1);
        }
    }
    return(0);
}

int blurBackground(cv::Mat& source, cv::Mat &destination)
{
    cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
    detectFaces(destination,rectVector);
    drawBoxes(destination,rectVector);
    cv::cvtColor(destination,destination,cv::COLOR_GRAY2BGR);
    blur5x5_2(source,destination);
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

int coloredFaces(cv::Mat& source, cv::Mat &destination)
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
    return 0;
}

int magnitude(cv::Mat &source, cv::Mat &destination )
{
    cv::Mat X = cv::Mat::zeros(source.rows,source.cols,CV_16SC3);
    cv::Mat Y = cv::Mat::zeros(source.rows,source.cols,CV_16SC3);
    cv::Mat M = cv::Mat::zeros(source.rows,source.cols/2,CV_16SC3);
    for(int i=1;i<source.rows-1;i++)
    {
        //define pointers for each channel in image
        cv::Vec3b *up = source.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *down = source.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *x=X.ptr<cv::Vec3b>(i);
        cv::Vec3b *y=Y.ptr<cv::Vec3b>(i);
        uchar *m=M.ptr<uchar>(i);
        //iterate through the each pixel and store blue, green and red values in respective variables.
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sumx=(-1*up[j-1][k])+(up[j+1][k])+
                        (-2*mid[j-1][k])+(2*mid[j+1][k])+
                        (-1*down[j-1][k])+(down[j+1][k]);
                int sumy=(1*up[j-1][k])+(2*up[j][k])+(up[j+1][k])-
                        (1*down[j-1][k])-(2*down[j][k])-(down[j+1][k]);
                sumx/=4;
                sumy/=4;
                x[j][k]=sumx;
                y[j][k]=sumy;
                m[j*3+k]=fmin(255,sqrt((sumx*sumx)+(sumy*sumy)));
                
            }  
        }
        destination=M;
    }
    return(0); 
}

int sobelX3x3(cv::Mat &source, cv::Mat &destination)
{   
    cv::Mat X = cv::Mat::zeros(source.rows,source.cols/2,CV_16SC3);
    for(int i=1;i<source.rows-1;i++)
    {
        cv::Vec3b *up = source.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *down = source.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *x=X.ptr<cv::Vec3b>(i);
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum=(-1*up[j-1][k])+(up[j+1][k])+
                        (-2*mid[j-1][k])+(2*mid[j+1][k])+
                        (-1*down[j-1][k])+(down[j+1][k]);
                //int sum=(-1*mid[j-1][k])+(mid[j+1][k]);
                sum/=4;
                x[j][k]=sum;
            }  
        } 
        // for(int j=1;j<source.cols-1;j++)
        // {
        //     for(int k=0;k<source.channels();k++)
        //     {
        //         int sum = X.at<cv::Vec3b>(i - 1,j)[k] +
        //                   (2 * X.at<cv::Vec3b>(i,j)[k])+ 
        //                   X.at<cv::Vec3b>(i + 1,j)[k];
        //         sum /= 4;
        //         X.at<cv::Vec3b>(i,j)[k] = sum;
        //     }  
        // }
        destination=X;  
    }
    return 0;
}

int sobelY3x3(cv::Mat &source, cv::Mat &destination)
{
    cv::Mat Y = cv::Mat::zeros(source.rows,source.cols/2,CV_16SC3);
    for(int i=1;i<source.rows-1;i++)
    {
        //define pointers for each channel in image
        cv::Vec3b *up = source.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *down = source.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *mid = source.ptr<cv::Vec3b>(i);
        cv::Vec3b *y=Y.ptr<cv::Vec3b>(i);
        //iterate through the each pixel and store blue, green and red values in respective variables.
        for(int j=1;j<source.cols-1;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum=(1*up[j-1][k])+(2*up[j][k])+(up[j+1][k])-
                        (1*down[j-1][k])-(2*down[j][k])-(down[j+1][k]);
                //int sum=(8*mid[j-1][k])+(16*mid[j][k])+(8*mid[j+1][k]);
                sum/=4;
                y[j][k]=sum;
                //printf("y[%i][%i][%i]=%i\n",i,j,k,sum);
            }  
        }
        destination=Y; 
    }
    // for(int i=1;i<source.rows-1;i++)
    // { 
    //     for(int j=1;j<source.cols-1;j++)
    //     {
    //         for(int k=0;k<source.channels();k++)
    //         {
    //             int sum = Y.at<cv::Vec3b>(i-1,j)[k]- Y.at<cv::Vec3b>(i + 1,j)[k];
    //             //sum/=4;
    //             //printf("Y.at<cv::Vec3b>(%i,%i)[%i]=%i\n",i,j,k,sum);
    //             Y.at<cv::Vec3b>(i,j)[k] = sum;
    //         }  
    //     }
    //     destination=Y;  
    // }
    
    return 0;
}

int blur5x5_1(cv::Mat &source, cv::Mat &destination)
{
    destination = cv::Mat::zeros(source.rows,source.cols,source.type());
    for(int i=2;i<source.rows-2;i++)
    {
        for(int j=2;j<source.cols-2;j++)
        {
            for(int k=0;k<source.channels();k++)
            {
                int sum=source.at<cv::Vec3b>(i-2,j-2)[k]+(2*source.at<cv::Vec3b>(i-2,j-1)[k])+(4*source.at<cv::Vec3b>(i-2,j)[k])+(2*source.at<cv::Vec3b>(i-2,j+1)[k])+(source.at<cv::Vec3b>(i-2,j+2)[k]);
                        (2*source.at<cv::Vec3b>(i-1,j-2)[k])+(4*source.at<cv::Vec3b>(i-1,j-1)[k])+(8*source.at<cv::Vec3b>(i-1,j)[k])+(4*source.at<cv::Vec3b>(i-1,j+1)[k])+(2*source.at<cv::Vec3b>(i-1,j+2)[k])+
                        (4*source.at<cv::Vec3b>(i,j-2)[k])+(8*source.at<cv::Vec3b>(i,j-1)[k])+(16*source.at<cv::Vec3b>(i,j)[k])+(8*source.at<cv::Vec3b>(i,j+1)[k])+(4*source.at<cv::Vec3b>(i,j+2)[k])+
                        (2*source.at<cv::Vec3b>(i+1,j-2)[k])+(4*source.at<cv::Vec3b>(i+1,j-1)[k])+(8*source.at<cv::Vec3b>(i+1,j)[k])+(4*source.at<cv::Vec3b>(i+1,j+1)[k])+(2*source.at<cv::Vec3b>(i+1,j+2)[k])+
                        (source.at<cv::Vec3b>(i+2,j-2)[k])+(2*source.at<cv::Vec3b>(i+2,j-1)[k])+(4*source.at<cv::Vec3b>(i+2,j)[k])+(2*source.at<cv::Vec3b>(i+2,j+1)[k])+(source.at<cv::Vec3b>(i+2,j+2)[k]);

                sum/=100;
                destination.at<cv::Vec3b>(i,j)[k]=sum;
            }
            
        }   
    }
    return 0;
}

int sepia( cv::Mat &source, cv::Mat &destination )
{
    cv::Mat sepia = cv::Mat::zeros(source.rows,source.cols,source.type());
    for(int i=0;i<source.rows;i++)
    {
        //define pointers for each channel in image
        uchar *o = source.ptr<uchar>(i);
        uchar *s = sepia.ptr<uchar>(i);
        //iterate through the each pixel and store blue, green and red values in respective variables.
        for(int j=0;j<source.cols;j++)
        {
            s[j*3]=fmin((o[j*3]*0.131)+(o[j*3+1]*0.534)+(o[j*3+2]*0.272),255);
            s[j*3+1]=fmin((o[j*3]*0.168)+(o[j*3+1]*0.686)+(o[j*3+2]*0.349),255);
            s[j*3+2]=fmin((o[j*3]*0.189)+(o[j*3+1]*0.769)+(o[j*3+2]*0.393),255);
        }   
    }  
    destination=sepia; 
    return 0;
}

int greyscale( cv::Mat &source, cv::Mat &destination )
{
    cv::Mat gray = cv::Mat::zeros(source.rows,source.cols,source.type());
    for(int i=0;i<source.rows;i++)
    {
        //define pointers for each channel in image
        uchar *o = source.ptr<uchar>(i);
        uchar *g = gray.ptr<uchar>(i);
        //iterate through the each pixel and store blue, green and red values in respective variables.
        for(int j=0;j<source.cols;j++)
        {
            g[j*3]=(o[j*3]-255);//used NTSC formula to convert BGR into grayscale
            g[j*3+1]=(o[j*3]-255);
            g[j*3+2]=(o[j*3]-255);
        }
    }   
    destination=gray; 
    return 0;
}

// int filter(int Flag,cv::Mat &source, cv::Mat &destination)
// {
    
//     if(Flag==1)
//     {
//         cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
//     }
//     else if(Flag==2)
//     {
//         destination=source;
//     }
//     else if(Flag==3)
//     {
//         greyscale(source,destination);   
//     }
//     else if(Flag==4)
//     {
//         sepia(source,destination);    
//     }
//     else if(Flag==5)
//     {
//         blur5x5_1(source,destination);   
//     }
//     else if(Flag==6)
//     {
//         blur5x5_2(source,destination);   
//     }
//     else if(Flag==7)
//     {
//         sobelX3x3(source,destination);   
//     }
//     else if(Flag==8)
//     {
//         sobelY3x3(source,destination);   
//     }
//     else if(Flag==9)
//     {
//         magnitude(source,destination);   
//     }
//     else if(Flag==10)
//     {
//         blurQuantize(source,destination,255);   
//     }
//     else if(Flag==11)
//     {
//         cv::cvtColor(source,destination,cv::COLOR_BGR2GRAY);
//         detectFaces(destination,rectVector);
//         drawBoxes(destination,rectVector);
//     }
//     else if(Flag==12)
//     {
//         coloredFaces(source,destination);
//     }
//     else if(Flag==13)
//     {
//         blurBackground(source,destination);
//     }
//     else if(Flag==14)
//     {
//         extraEffects3(source,destination);
//     }
//     else
//     {
//         destination=source;
//     }
    
//     return 0;
// }

// int image_modification(cv::Mat &source,cv::Mat &destination)
// {   
//     if(cv::waitKey(1)=='s')
//     {
//         printf("save");
//         while(i<50)
//         {
//             std::string j=std::to_string(i);
//             cv::imwrite("frame"+j+".jpeg",source);
//             i+=1;
//             break;
//         }
//     }
//     if(cv::waitKey(1)=='g')
//     {   
//         flag=1;
//     }
//     if(cv::waitKey(1)=='o')
//     {
//         flag=2;
//     }
//     if(cv::waitKey(1)=='h')
//     {
//         flag=3;
//     }
//     if(cv::waitKey(1)=='e')
//     {
//         flag=4;
//     }
//     if(cv::waitKey(1)=='b')
//     {
//         flag=5;
//     }
//     if(cv::waitKey(1)=='c')
//     {
//         flag=6;
//     }
//     if(cv::waitKey(1)=='x')
//     {
//         flag=7;
//     }
//     if(cv::waitKey(1)=='y')
//     {
//         flag=8;
//     }
//     if(cv::waitKey(1)=='m')
//     {
//         flag=9;
//     }
//     if(cv::waitKey(1)=='l')
//     {
//         flag=10;
//     }
//     if(cv::waitKey(1)=='f')
//     {
//         flag=11;
//     }
//     if(cv::waitKey(1)=='j')
//     {
//         flag=12;
//     }
//     if(cv::waitKey(1)=='v')
//     {
//         flag=13;
//     }
//     if(cv::waitKey(1)=='a')
//     {
//         flag=14;
//     }
//     filter(flag,source,destination);
//     return 0;
// }