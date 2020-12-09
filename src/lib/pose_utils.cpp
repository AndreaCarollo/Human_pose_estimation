#include "pose_utils.h"
using namespace std; // standard
using namespace cv;  // openCV
using namespace dnn; // deep neural network

#define PI 3.14159265

// void human_pose::extract_keypoints(cv::Mat result)
// {

//     H = result.size[2];
//     W = result.size[3];

//     // find the position of the body parts
//     vector<Point> points(22);
//     for (int n = 0; n < nparts; n++)
//     {
//         // Slice heatmap of corresponding body's part.
//         Mat heatMap(H, W, CV_32F, result.ptr(0, n));
//         // 1 maximum per heatmap
//         Point p(-1, -1), pm;
//         double conf;
//         minMaxLoc(heatMap, 0, &conf, 0, &pm);
//         if (conf > thresh)
//             p = pm;
//         points[n] = p;
//     }
//     keypoints = points;
// };

// cv::Point human_pose::get_body_kp(int idx)
// {
//     return keypoints[idx];
// };

// void human_pose::draw_pose(cv::Mat *display_img)
// {
//     // connect body parts and draw it !
//     float SX = float(display_img->cols) / W;
//     float SY = float(display_img->rows) / H;
//     // cout << " n pairs: " << npairs << "  --  n parts:" << nparts << endl;
//     for (int n = 0; n < npairs; n++)
//     {
//         // lookup 2 connected body/hand parts
//         Point2f a = keypoints[posePairs[n].first];  // std::cout << "a " << a << " ";
//         Point2f b = keypoints[posePairs[n].second]; // std::cout << "b " << b << " ";
//         // we did not find enough confidence before
//         if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
//             continue;
//         // scale to image size
//         a.x *= SX;
//         a.y *= SY;
//         b.x *= SX;
//         b.y *= SY;
//         cv::line((*display_img), a, b, colors_openpose[n], 2);
//         cv::circle((*display_img), a, 3, Scalar(0, 0, 200), -1);
//         cv::circle((*display_img), b, 3, Scalar(0, 0, 200), -1);
//     }
// };

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
    return (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
}

float distance(cv::Point A, cv::Point B)
{

    // Calculate the distance between points A and B
    return sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
}

//  ___                  _    _
// | __| _  _  _ _   __ | |_ (_) ___  _ _   ___
// | _| | || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

cv::RotatedRect getErrorEllipse(float chisquare_val, cv::Point2f mean, cv::Mat covmat)
{

    //Get the eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covmat, eigenvalues, eigenvectors);
    // cout << "eigen values : \n" << eigenvalues.at<float>(0) << "\n" << eigenvalues.at<float>(1) << endl;

    //Calculate the angle between the largest eigenvector and the x-axis
    float angle = atan2(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(0, 0));

    //Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if (angle < 0)
        angle += 6.28318530718;

    //Conver to degrees instead of radians
    angle = 180 * angle / 3.14159265359;

    //Calculate the size of the minor and major axes
    float halfmajoraxissize = chisquare_val * sqrt(eigenvalues.at<float>(0));
    float halfminoraxissize = chisquare_val * sqrt(eigenvalues.at<float>(1));

    //Return the oriented ellipse
    //The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
    return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);
}

cv::RotatedRect generate_cov_ellipse(cv::Mat heatMap, vector<Point> contour, cv::Point2f mean)
{
    cv::RotatedRect ellipse;

    vector<Point> contour_poly;
    cv::approxPolyDP(contour, contour_poly, 3, true);
    cv::Rect contour_rect = cv::boundingRect(contour);

    if (contour_rect.empty())
    {
        return ellipse;
    }

    // evalaute the means value for each row and col
    std::vector<double> vector_x;
    std::vector<double> vector_y;
    std::vector<double> vector_p;
    for (int m = contour_rect.tl().x; m < contour_rect.tl().x + contour_rect.width; m++)
    {
        for (int n = contour_rect.tl().y; n < contour_rect.tl().y + contour_rect.height; n++)
        {
            if (cv::pointPolygonTest(contour, cv::Point(m, n), false) >= 0)
            {
                vector_x.push_back(m);
                vector_y.push_back(n);
                vector_p.push_back(heatMap.at<double>(m, n));
            }
        }
    }

    // expected value
    float x_mean = 0.0;
    float y_mean = 0.0;
    float p_tot = 0.0;
    for (int k = 0; k < vector_x.size(); k++)
    {
        x_mean += vector_x[k] * vector_p[k];
        y_mean += vector_y[k] * vector_p[k];
        p_tot += vector_p[k];
    }
    x_mean /= p_tot;
    y_mean /= p_tot;

    // evaluate covariance
    float cov_xx = 0;
    float cov_yy = 0;
    float cov_xy = 0;
    for (int k = 0; k < vector_x.size(); k++)
    {
        cov_xx += (vector_x[k] - x_mean) * (vector_x[k] - x_mean) * vector_p[k];
        cov_yy += (vector_y[k] - y_mean) * (vector_y[k] - y_mean) * vector_p[k];
        cov_xy += (vector_x[k] - x_mean) * (vector_y[k] - y_mean) * vector_p[k];
    }
    cov_xx /= p_tot;
    cov_yy /= p_tot;
    cov_xy /= p_tot;
    // covariance matrix
    float data[4] = {cov_xx, cov_xy, cov_xy, cov_yy};
    cv::Mat cov_matrix = cv::Mat(2, 2, CV_32F, data);
    // std::cout << n << " - cov Matrix : " << cov_matrix << std::endl;
    if(mean!=cv::Point2f(-1,-1)){
        x_mean = mean.x;
        y_mean = mean.y;
    }

    //Calculate the error ellipse for a 95% confidence interval ( 2.4477 )
    ellipse = getErrorEllipse(2.4477, cv::Point(x_mean, y_mean), cov_matrix);
    return ellipse;
}

//   ___           _                          _              _
//  / __| ___  ___| |_  _  _  _ _  ___    __ | |_   ___  __ | |__
// | (_ |/ -_)(_-<|  _|| || || '_|/ -_)  / _|| ' \ / -_)/ _|| / /
//  \___|\___|/__/ \__| \_,_||_|  \___|  \__||_||_|\___|\__||_\_\
//
