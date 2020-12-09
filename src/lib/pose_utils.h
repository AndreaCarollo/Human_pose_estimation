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

cv::RotatedRect getErrorEllipse(float chisquare_val, cv::Point2f mean, cv::Mat covmat);

cv::RotatedRect generate_cov_ellipse(cv::Mat heatMap, vector<Point> contour, cv::Point2f mean = cv::Point2f(-1,-1));



#endif