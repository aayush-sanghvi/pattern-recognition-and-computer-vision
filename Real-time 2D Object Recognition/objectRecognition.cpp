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
#include "imageProcessing.h"
#include "csv_util.h"

// Main function
int main(int argc, char *argv[])
{   
    // Declare variables to store different frames of video, vectors, Moments and boolean.
    cv::Mat threshold;// Threshold image
    cv::Mat clean; //Cleaned up image
    cv::Mat seg; // Segmented image
    cv::Mat liveVideo; // Live video recorded from camera
    cv::Mat regions; //Image to display different regions in the video frame
    cv::Mat centroids; // Matrix to hold centroids of each region
    cv::Moments m; // Variable to hold moments of each region
    char db[256]= "data.csv";
    bool training = false;
    std::vector<int> largeLabels; // Vector to hold top 3 largest region present in video frame
    std::vector<std::string> classNamesDB;// Vector to hold name of each region during training
    std::vector<std::vector<float>> featuresDB;// Vector to hold moments of each region during training

    // Create a window named "Feature" to display different objects in  video frame.
    cv::namedWindow("Feature",cv::WINDOW_NORMAL);

    // Save IP Address and port number for DroidCam application.
    std::string ipAddress = "10.110.14.136";
    int port = 4747;

    // URL for connecting to DroidCam
    std::string url = "http://" + ipAddress + ":" + std::to_string(port) + "/video";
    
    // Reads the video and stores it in opencv VideoCapture datatype variable.
    cv::VideoCapture cap(url);
    
    // Checks if the video capture has been initialise and captured video is a valid video.
    if(!cap.isOpened())
    {
        std::cout<<"Could not open video";
        exit(-1);
    }
        
    // Creates an infinite loop.
    while(1)
    {
        // Stores a new frame from the video into a variable defined above.
        cap>>liveVideo;

        // See if there is a waiting keystroke for the video
        char key = cv::waitKey(100); 
        
        // Switch between training mode and inference mode
        if (key == 't') {
            training = true;
            std::cout << "Training Mode" << std::endl;
        }
        else{
            training = false;
        }

        // Execute feature function to get thresholded image, cleaned-up image, segmented images,
        // and stats of each regions such as top 3 largest regions and labels, centroid and 
        // moments of each region.
        std::vector<std::vector<float>> vec=feature(liveVideo,threshold,clean,seg,regions,m,largeLabels,centroids);
        

        // Iterate through top 3 largest region detected in a live video.
        for(int n=0;n<largeLabels.size();n++){

            // Create a matrix to store first region detected in live video.
            cv::Mat region=(regions==largeLabels[n]);
            
            //Training mode.
            if (training) {

                //-----Labelling---
                // Display current region in binary form
                cv::namedWindow("Current Region", cv::WINDOW_AUTOSIZE);
                cv::imshow("Current Region", region);

                // Ask the user for a class name
                std::cout << "Input the class for this object." << std::endl;

                // Monitors key press to identify the object.
                char k = cv::waitKey(0);
                cv::destroyWindow("Current Region");

                // Executes getCLassName function to look into the Look-up 
                // table and saves the object name in a variable
                char* className = getClassName(k);
                printf("%s\n",className);

                append_image_data_csv(db, className,vec[n] );
                // Ends training mode once all regions are identified and closes the region window
            }

            // Inference Mode
            else{
                // Calculates and compares the distance metrics of current region in live video with Hu Moments 
                // of recognised objects and places a name on the region.
                std::vector<std::vector<float>> chan_csv;
                std::vector<char*> classNamesDB;
                read_image_data_csv(db,classNamesDB,chan_csv);
                
                std::string className= classifier(chan_csv, classNamesDB, vec[n]);
                cv::putText(liveVideo, className, cv::Point(centroids.at<double>(largeLabels[n], 0),centroids.at<double>(largeLabels[n], 1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                //Calculates and compares the distance metrics of current region in live video with Hu Moments 
                // of recognised objects and returns a name on the region to the neareast k neighbours
                //std::vector<std::string> names = kNN(featuresDB, classNamesDB, huMoments, 2);
            }

        }

        
        // Displays frames with recognised objects and rectangle placed detecting the boundary of each object.
        cv::imshow("Segmented",seg);
        cv::imshow("Feature",liveVideo);

        //checks for 'q' key press by the user to exit the program.
        if(cv::waitKey(1)=='q')
        {
            break;
        }
    }

    //releases the video and terminates the window created.
    liveVideo.release();
    cv::destroyAllWindows();

    //ends the main function by returning 0 value to main program.
    return 0;
}