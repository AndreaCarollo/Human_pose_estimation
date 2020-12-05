#include "pose_utils.h"
using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

#define PI 3.14159265

void human_pose::extract_keypoints(cv::Mat result)
{

    H = result.size[2];
    W = result.size[3];

    // find the position of the body parts
    vector<Point> points(22);
    for (int n = 0; n < nparts; n++)
    {
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        // 1 maximum per heatmap
        Point p(-1, -1), pm;
        double conf;
        minMaxLoc(heatMap, 0, &conf, 0, &pm);
        if (conf > thresh)
            p = pm;
        points[n] = p;
    }
    keypoints = points;
};

cv::Point human_pose::get_body_kp(int idx)
{
    return keypoints[idx];
};

void human_pose::draw_pose(cv::Mat *display_img)
{
    // connect body parts and draw it !
    float SX = float(display_img->cols) / W;
    float SY = float(display_img->rows) / H;
    // cout << " n pairs: " << npairs << "  --  n parts:" << nparts << endl;
    for (int n = 0; n < npairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f a = keypoints[posePairs[n].first];  // std::cout << "a " << a << " ";
        Point2f b = keypoints[posePairs[n].second]; // std::cout << "b " << b << " ";
        // we did not find enough confidence before
        if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
            continue;
        // scale to image size
        a.x *= SX;
        a.y *= SY;
        b.x *= SX;
        b.y *= SY;
        cv::line((*display_img), a, b, colors_openpose[n], 2);
        cv::circle((*display_img), a, 3, Scalar(0, 0, 200), -1);
        cv::circle((*display_img), b, 3, Scalar(0, 0, 200), -1);
    }
};


//  _                          _        _                _    _  _
// | |__ ___  _  _  _ __  ___ (_) _ _  | |_  ___   _  _ | |_ (_)| | ___
// | / // -_)| || || '_ \/ _ \| || ' \ |  _|(_-<  | || ||  _|| || |(_-<
// |_\_\\___| \_, || .__/\___/|_||_||_| \__|/__/   \_,_| \__||_||_|/__/
//            |__/ |_|
//
bool ccw(cv::Point A, cv::Point B, cv::Point C)
{
    /* Returns True if the 3 points A,
        B and C are listed in a counterclockwise order
        ie if the slope of the line AB is less than the slope of AC
        https : //bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    */
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

bool intersect(cv::Point A, cv::Point B, cv::Point C, cv::Point D)
{
    /*
        Return true if line segments AB and CD intersect
        https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    */
    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0 or C.x < 0 or C.y < 0 or D.x < 0 or D.y < 0)
    {
        return false;
    }
    else
    {
        return ccw(A, C, D) != ccw(B, C, D) & ccw(A, B, C) != ccw(A, B, D);
    }
}

float angle(cv::Point A, cv::Point B, cv::Point C)
{
    //* Calculate the angle between segment(A,B) and segment (B,C)

    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0 or C.x < 0 or C.y < 0)
    {
        std::cout << "error input" << std::endl;
        return 0;
    }
    return (atan2(C.y - B.y, C.x - B.x) - atan2(A.y - B.y, A.x - B.x)) * 180 / PI;
}

float vertical_angle(cv::Point A, cv::Point B)
{
    //* Calculate the angle between segment(A,B) and vertical axe
    if (A.x < 0 or A.y < 0 or B.x < 0 or B.y < 0)
    {
        std::cout << "error input" << std::endl;
        return 0;
    }
    else
    {
        return (atan2(B.y - A.y, B.x - A.x) - PI / 2.0) * 180 / PI;
    }
}

float sq_distance(cv::Point A, cv::Point B)
{

    // Calculate the square of the distance between points A and B
    return (B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y);
}

float distance(cv::Point A, cv::Point B)
{

    // Calculate the distance between points A and B
    return sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
}

//   ___           _                          _              _
//  / __| ___  ___| |_  _  _  _ _  ___    __ | |_   ___  __ | |__
// | (_ |/ -_)(_-<|  _|| || || '_|/ -_)  / _|| ' \ / -_)/ _|| / /
//  \___|\___|/__/ \__| \_,_||_|  \___|  \__||_||_|\___|\__||_\_\
//

#define RESET '\0'
#define HANDS_UP '\1'
#define HORIZONTAL '\2'
#define HANDS_ON_NECK '\4'

bool human_pose::gesture_activation()
{
    Point nose = get_body_kp(NOSE);
    Point neck = get_body_kp(NECK);
    Point r_wrist = get_body_kp(RIGHT_WRIST);
    Point l_wrist = get_body_kp(LEFT_WRIST);
    Point r_elbow = get_body_kp(RIGHT_ELBOW);
    Point l_elbow = get_body_kp(LEFT_ELBOW);
    Point r_shoulder = get_body_kp(RIGHT_SHOULDER);
    Point l_shoulder = get_body_kp(LEFT_SHOULDER);
    bool flag_shoulder = l_shoulder.x < 0 & l_shoulder.y < 0 & r_shoulder.x < 0 & r_shoulder.y < 0;
    bool flag_elbow = r_elbow.x < 0 & r_elbow.y < 0 & l_elbow.x < 0 & l_elbow.y < 0;
    bool flag_wrist = r_wrist.x < 0 & r_wrist.y < 0 & l_wrist.x < 0 & l_wrist.y < 0;
    bool flag_neck = neck.x < 0 & neck.y < 0;
    bool flag_nose = nose.x < 0 & nose.y < 0;
    if (flag_shoulder | flag_elbow | flag_wrist | flag_neck | flag_nose)
    {
        gesture = RESET;
        std::cout << "missed point" << endl;
        return false;
    }
    else // check the gesture pose
    {
        // evaluate the shoulder width
        double shoulder_width = sq_distance(l_shoulder, r_shoulder);
        double hands_distance = sq_distance(l_wrist, r_wrist);
        cout << "shoulder dist:  " << shoulder_width << " -- hands dist:  " << hands_distance << endl;

        // evaluate arm angles
        double vertical_angle_right_arm = vertical_angle(r_wrist, r_elbow);
        double vertical_angle_left_arm = vertical_angle(l_wrist, l_elbow);

        bool left_hand_up = l_wrist.y < (neck.y + nose.y) / 2;
        bool right_hand_up = r_wrist.y < (neck.y + nose.y) / 2;
        bool large_distance_hands = hands_distance > shoulder_width;
        if (left_hand_up && right_hand_up && large_distance_hands)
        {
            cout << "ACTIVATION GESTURE" << endl;
            return true;
        }
        else
        {
            return false;
        }
    }
};

void human_pose::gesture_check()
{
    gesture = RESET;
    // extract usefull keypoints
    Point nose = get_body_kp(NOSE);
    Point neck = get_body_kp(NECK);
    Point r_wrist = get_body_kp(RIGHT_WRIST);
    Point l_wrist = get_body_kp(LEFT_WRIST);
    Point r_elbow = get_body_kp(RIGHT_ELBOW);
    Point l_elbow = get_body_kp(LEFT_ELBOW);
    Point r_shoulder = get_body_kp(RIGHT_SHOULDER);
    Point l_shoulder = get_body_kp(LEFT_SHOULDER);

    // check if we have the shoulder points
    // (Note: from the extraction skeleton, the not finded poits has -1 coordinates)
    bool flag_shoulder = l_shoulder.x < 0 & l_shoulder.y < 0 & r_shoulder.x < 0 & r_shoulder.y < 0;
    bool flag_elbow = r_elbow.x < 0 & r_elbow.y < 0 & l_elbow.x < 0 & l_elbow.y < 0;
    bool flag_wrist = r_wrist.x < 0 & r_wrist.y < 0 & l_wrist.x < 0 & l_wrist.y < 0;
    bool flag_nose_neck = nose.x < 0 & nose.y < 0 & neck.x < 0 & neck.y < 0;
    if (flag_shoulder | flag_elbow | flag_wrist | flag_nose_neck)
    {
        gesture = RESET;
        std::cout << "missed point" << endl;
        return;
    }
    else // check the gesture pose
    {
        // evaluate the shoulder width
        double shoulder_width = distance(l_shoulder, r_shoulder);
        double hands_distance = distance(l_wrist, r_wrist);

        // evaluate arm angles
        double vertical_angle_right_arm = vertical_angle(r_wrist, r_elbow);
        double vertical_angle_left_arm = vertical_angle(l_wrist, l_elbow);

        bool left_hand_up = l_wrist.y < neck.y;
        bool right_hand_up = r_wrist.y < neck.y;
        if (left_hand_up & right_hand_up) //& hands_distance >= shoulder_width)
        {
            cout << "ACTIVATION GESTURE" << endl;
        }
    }
};

///////////

