/*
Aayush H Sanghvi
Spring 2024 semseter
Date:- 21st Jan 2024
CS5330- Pattern Recognition and Computer Vision.

To display image and apply different filters 
using the OpenCV library on that image.
*/

//Includes all the necessay libraries required.
#include <iostream>
#include <sys/time.h>
#include <opencv4/opencv2/opencv.hpp>
#include "filter.h"

// int blur5x5_1( cv::Mat &src, cv::Mat &dst );
// int blur5x5_2( cv::Mat &src, cv::Mat &dst );


double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
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
    cv::Mat src = cv::imread((std::string)argv[1]);
    cv::Mat dst;

    // checks if the image read is a valid image (not empty).
    if(src.empty())
    {
        printf("Not a valid image. Please read a valid image");
        exit(-1);
    }

    const int Ntimes = 10;
	
  //////////////////////////////
  // set up the timing for version 1
  double startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_1( src, dst );
  }

  // end the timing
  double endTime = getTime();

  // compute the time per image
  double difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (1): %.4lf seconds for blur5x5_1 function\n", difference );

  //////////////////////////////
  // set up the timing for version 2
  startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_2( src, dst );
  }

  // end the timing
  endTime = getTime();

  // compute the time per image
  difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (2): %.4lf seconds for blur5x5_2 function\n", difference );
    //creates a named window and displays the image in window using imshow function.
    cv::namedWindow(argv[1],1);
    cv::imshow(argv[1],src);

    //checks for key press by the user to either to exit the program or display different colors of the image.
    while(1)
    {
        //if 'q' key is pressed, the program exits
        if(cv::waitKey(0)=='q')
        {
           break;
        }
    }

    //ends the main function by returning 0 value to main program.
    return 0;
}