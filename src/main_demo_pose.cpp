// my libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include "./lib/dnn_loader.h"
#include "./lib/pose_extractor.h"

using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

int main()
{

    string video_path = "";
    cv::VideoCapture cap(-1);

    if (!cap.isOpened())
    {
        std::cout << "error opening camera" << std::endl;
        return 0;
    }

    cv::Mat img, display_img;

    //_______ POSE ESTIMATION: load neural network and config ______________________________________
    cv::dnn::Net network_pose;

    // std::string model_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml";
    // std::string config_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin";
    std::string model_POSE = "path/to/the/model/human-pose-estimation-0001.xml";
    std::string config_POSE = "path/to/the/weigth/human-pose-estimation-0001.bin";

    load_net_pose(&network_pose, model_POSE, config_POSE);

    cv::Rect parse_box;

    std::vector<cv::Rect> DETECTION_people_in_frame, DETECTION_faces_in_frame, REID_people_in_frame;
    std::vector<float> REID_people_feature_in_frame;

    bool flag_fail_update_tracker = false;

    bool plot_kalman = false;

    // Start and end times
    time_t start, end;

    // Start time
    time(&start);
    double count_frame = 0;

    bool parse_pose = false;

    for (;;)
    {
        cap >> img;
        // copy frame for plot & show
        img.copyTo(display_img);

        // >>>>> Pose Extraction
        std::vector<KeyPoint_pose> keyPointsList;
        // poses contains vectors of index of the point inside \keyPointsList
        std::vector<std::vector<int>> poses = get_poses(&img, &network_pose, &keyPointsList, 2, true);

        // given a rectangle, extract the body pose inside it
        std::vector<int> User_pose = nullpose;
        if (parse_pose)
        {
            User_pose = parse_poses(&parse_box,
                                    keyPointsList, &poses, &display_img);
        }

        plot_skeleton(&display_img, User_pose, keyPointsList);
        plot_all_skeleton(&display_img, poses, keyPointsList, true);

        // ------- Display ------------------------------------------------------
        cv::namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
        cv::imshow("Display window", display_img);
        int k = waitKey(5);
        if (k == 27) // or count_frame == 200)
        {
            break;
        }
    }

    return 0;
}
