#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>

#include "pose_basic.h"

#ifndef POSE_UTILS
#define POSE_UTILS

using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

// Class human pose (collector of points and data)
class human_pose
{
private:
    /* data */
    int npairs = 17;
    int nparts = 18;
    float thresh = 0.2;

public:
    cv::Mat HeatMap;
    cv::Mat Pafs;
    std::vector<cv::Point> keypoints;
    std::vector<std::vector<cv::Point>> all_keypoints;
    std::vector<cv::Point3d> keypoints_3D;

    int H, W; // need for extract keypoints & draw pose

    // bit gesture
    // HANDS_UP '\1'

    uint8_t gesture;

    // Given the matrix output of the network, extract the point for 1 body
    // >> TODO: modify in order to get all the skeleton in the frame
    void extract_keypoints(cv::Mat result);
    void extract_all_keypoints(cv::Mat result);

    // Given the 2D points on the RGB frame, obtain the 3rd coordinate from the depth map
    // and convert all the pose in real measurement -> meters (not pixels)
    // >> TODO: get point depth from realsense
    void extract_keypoints_3D(); // using the depth + informations of scale to real measure

    cv::Point get_body_kp(int idx); // save points from the vector to the class

    // return true if the operator rises both the hands over the shoulders
    // return false if not or if the detection missed some necessary keypoints
    bool gesture_activation();
    void gesture_check();

    // draw pose to image
    void draw_pose(cv::Mat *display_img);
};


//  _                          _        _                _    _  _
// | |__ ___  _  _  _ __  ___ (_) _ _  | |_  ___   _  _ | |_ (_)| | ___
// | / // -_)| || || '_ \/ _ \| || ' \ |  _|(_-<  | || ||  _|| || |(_-<
// |_\_\\___| \_, || .__/\___/|_||_||_| \__|/__/   \_,_| \__||_||_|/__/
//            |__/ |_|
//

// Returns True if the 3 points A,
// B and C are listed in a counterclockwise order
// ie if the slope of the line AB is less than the slope of AC
// https : //bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
bool ccw(cv::Point A, cv::Point B, cv::Point C);

// Return true if line segments AB and CD intersect
// https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
bool intersect(cv::Point A, cv::Point B, cv::Point C, cv::Point D);

// Calculate the angle between segment(A,B) and segment (B,C)
float angle(cv::Point A, cv::Point B, cv::Point C);

// Calculate the angle between segment(A,B) and vertical axe
float vertical_angle(cv::Point A, cv::Point B);

// Calculate the square of the distance between points A and B
float sq_distance(cv::Point A, cv::Point B);

// Calculate the distance between points A and B
float distance(cv::Point A, cv::Point B);




//  ___                  _    _                
// | __| _  _  _ _   __ | |_ (_) ___  _ _   ___
// | _| | || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//                                          

#endif