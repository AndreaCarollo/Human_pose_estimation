// my libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/optim.hpp>

#include "./lib/dnn_loader.h"
#include "./lib/pose_extractor.h"

using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

const cv::Scalar color_plot = cv::Scalar(0, 0, 250);

int main()
{
    cv::ocl::setUseOpenCL(true);

    string video_path = "/dev/v4l/by-id/usb-046d_0825_0C52A0E0-video-index0";

    // cv::VideoCapture cap("../group.jpg");
    // cv::VideoCapture cap(0);
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened())
    {
        std::cout << "error opening camera" << std::endl;
        return 0;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 0);

    cv::Mat img, display_img;

    //_______ POSE ESTIMATION: load neural network and config ______________________________________
    cv::dnn::Net network_pose;

    std::string model_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml";
    std::string config_POSE = "/home/andrea/openvino_models/ir/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin";
    // std::string model_POSE = "path/to/the/model/human-pose-estimation-0001.xml";
    // std::string config_POSE = "path/to/the/weigth/human-pose-estimation-0001.bin";

    // load_net_pose(&network_pose, model_POSE, config_POSE);
    network_pose = readNetFromModelOptimizer(model_POSE, config_POSE);

    //* Set backend and target to execute network
    network_pose.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    network_pose.setPreferableTarget(DNN_TARGET_OPENCL);

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
        std::vector<std::vector<int>> poses = get_poses(&img, &network_pose, &keyPointsList, 1.75, true, 5);

        // given a rectangle, extract the body pose inside it
        std::vector<int> User_pose = nullpose;
        if (parse_pose)
        {
            User_pose = parse_poses(&parse_box,
                                    keyPointsList, &poses, &display_img);
        }

        // plot_all_skeleton(&display_img, poses, keyPointsList, true);
        plot_all_skeleton_single_color(&display_img, poses, keyPointsList, color_plot);

        // for (int k = 0; k < poses.size(); k++)
        // {
        //     auto index_hand = poses[k][RIGHT_WRIST];
        //     if (index_hand != -1)
        //     {
        //         cout << k << " -- hand position " << keyPointsList[index_hand].point << " -- ellipse size " << keyPointsList[index_hand].ellipse.size << endl;
        //         try
        //         {
        //             cv::ellipse(display_img, keyPointsList[index_hand].ellipse, cv::Scalar(0, 0, 255), 1, 8);
        //         }
        //         catch (const std::exception &e)
        //         {
        //             // std::cerr << e.what() << '\n';
        //         }
        //     }
        // }

        // ------- Display ---------------------------------------------------------------
        cv::namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
        cv::imshow("Display window", display_img);
        int k = waitKey(10);
        if (k == 27) // or count_frame == 200)
        {
            break;
        }
    }

    return 0;
}
