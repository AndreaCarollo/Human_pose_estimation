#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>

#ifndef UTILS_LIB
#define UTILS_LIB

enum StateMachine
{
    DETECT,
    TRACK,
    REIDENTIFICATION
};

struct parameters
{
    /* data */
    // middle detection windows in percentage
    float detWith = 0.25;
    // image size
    int imgCols, imgRows;

    // scale depth camera
    double scale_depth;

    StateMachine current_state = DETECT;
    StateMachine previous_state = DETECT;


};


// compute the cosine similarity of two vector
double cosine_similarity(std::vector<float> A, std::vector<float> B);

std::vector<float> extract_face_feature(cv::dnn::Net net_reid, cv::Mat img);

std::vector<float> extract_body_feature(cv::dnn::Net *net_reid, cv::Mat img);


cv::Point getCentre_rect(cv::Rect ROI);

void resizeRect(cv::Rect *oggetto, int cols, int rows);

void resizeRects(std::vector<cv::Rect> *oggetti, int cols, int rows);

void rescaleRect(cv::Rect2d *rettangolo, float scala);

bool isOverlap(cv::Rect ROI_1, cv::Rect ROI_2, bool is_face);

float scoreOverlap(cv::Rect ROI_1, cv::Rect ROI_2);

bool findHisBody(std::vector<cv::Rect> bodies, cv::Rect face, cv::Rect *output); //TrackableObject *personToTrack)


std::vector<cv::Rect> DetectionPedestrian_on_frame(cv::Mat img, cv::dnn::Net *network);

std::vector<cv::Rect> DetectionFaces_on_frame(cv::Mat img, cv::dnn::Net *network);




#endif