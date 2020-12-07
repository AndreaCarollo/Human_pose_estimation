#include "./pose_basic.h"

//   __                  _    _
//  / _| _  _  _ _   __ | |_ (_) ___  _ _   ___
// |  _|| || || ' \ / _||  _|| |/ _ \| ' \ (_-<
// |_|   \_,_||_||_|\__| \__||_|\___/|_||_|/__/
//

void extract_all_keypoints(cv::Mat result, cv::Size targetSize,
                           vector<vector<KeyPoint_pose>> *all_keyPoints,
                           std::vector<KeyPoint_pose> *keyPointsList)
{

    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;
    float thresh = 0.12;
    int keyPointId = 0;
    cv::Size blur_size = cv::Size(3, 3);
    // vector<vector<KeyPoint_pose>> all_keyPoints[18];
    // find the position of the body parts
    for (int n = 0; n < nparts; n++)
    {
        vector<KeyPoint_pose> keyPoints;
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0, n));
        cv::UMat UheatMap;
        heatMap.copyTo(UheatMap);
        // cv::UMat UresizedHeatMap;
        // cv::resize(UheatMap, UresizedHeatMap, targetSize);
        cv::resize(UheatMap, UheatMap, targetSize);

        // cv::UMat UsmoothProbMap;
        // cv::GaussianBlur(UresizedHeatMap, UsmoothProbMap, blur_size, 0, 0);
        // cv::GaussianBlur(UheatMap, UheatMap, blur_size, 0, 0);
        UheatMap.copyTo(heatMap);

        // cv::Mat smoothProbMap;
        // UsmoothProbMap.copyTo(smoothProbMap);

        cv::UMat UmaskedProbMap;
        // cv::threshold(UsmoothProbMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        cv::threshold(UheatMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        // cv::Mat maskedProbMap;
        // UmaskedProbMap.copyTo(maskedProbMap);

        // maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);
        UmaskedProbMap.convertTo(UmaskedProbMap, CV_8U, 1);

        std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::findContours(UmaskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); ++i)
        {
            // cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());
            cv::UMat blobMask = cv::UMat::zeros(UheatMap.rows, UheatMap.cols, UheatMap.type());

            cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

            double maxVal;
            cv::Point maxLoc;

            // cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);
            cv::minMaxLoc(UheatMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

            KeyPoint_pose tmp_point = KeyPoint_pose(maxLoc, heatMap.at<float>(maxLoc.y, maxLoc.x));
            if (contours[i].size() > 5)
            {
                cv::RotatedRect ellipse_contour = fitEllipse(contours[i]);
                tmp_point.ellipse = ellipse_contour;
            }
            keyPoints.push_back(tmp_point);
        }

        // give a unic id number to all points
        for (int i = 0; i < keyPoints.size(); i++, keyPointId++)
        {
            keyPoints[i].id = keyPointId;
        }

        all_keyPoints->push_back(keyPoints);
        keyPointsList->insert(keyPointsList->end(), keyPoints.begin(), keyPoints.end());
    };
};

void extract_all_keypoints_test(cv::Mat result, cv::Size targetSize,
                                vector<vector<KeyPoint_pose>> *all_keyPoints,
                                std::vector<KeyPoint_pose> *keyPointsList, cv::Mat *img, bool plot_flag)
{

    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;
    float thresh = 0.1;
    int keyPointId = 0;
    cv::Size blur_size = cv::Size(3, 3);
    // vector<vector<KeyPoint_pose>> all_keyPoints[18];
    // find the position of the body parts
    std::vector<RotatedRect> ellissi;
    std::vector<Point> punti;
    for (int n = 0; n < nparts; n++)
    {
        vector<KeyPoint_pose> keyPoints;
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32FC1, result.ptr(0, n));
        cv::UMat UheatMap;
        heatMap.copyTo(UheatMap);
        cv::UMat UresizedHeatMap;
        cv::resize(UheatMap, UresizedHeatMap, targetSize);

        cv::UMat UsmoothProbMap;
        cv::GaussianBlur(UresizedHeatMap, UsmoothProbMap, blur_size, 0, 0);

        cv::UMat UmaskedProbMap;
        cv::threshold(UsmoothProbMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        cv::Mat maskedProbMap;
        UmaskedProbMap.copyTo(maskedProbMap);

        maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++)
        {

            if (n == NOSE || n == NECK || n == LEFT_EAR || n == LEFT_EYE || n == RIGHT_EAR || n == RIGHT_EYE)
            {
                if (contours[i].size() < 10)
                {
                    continue;
                }
            }
            else
            {
                if (contours[i].size() < 20)
                {
                    continue;
                }
            }
            cv::RotatedRect ellipse_contour = fitEllipse(contours[i]);
            ellissi.push_back(ellipse_contour);
            //* plot ellipse
            //cv::ellipse(img_display, ellipse_contour, cv::Scalar(255, 0, 0), 1, 8);
            cv::Rect ellipse_rect = ellipse_contour.boundingRect();
            auto ellipse_a = ellipse_contour.size.width / 2;
            auto ellipse_b = ellipse_contour.size.height / 2;

            auto ellipse_centre = ellipse_contour.center;
            auto ellipse_angle = ellipse_contour.angle;
            auto tl = ellipse_rect.tl();

            float x_keypoint = 0;
            float y_keypoint = 0;
            float prob_keypoint = 0;
            int counter = 0;

            for (int i = 0; i < ellipse_rect.width; i++)
            {
                /* code */
                for (int k = 0; k < ellipse_rect.height; k++)
                {
                    /* code */
                    auto point = tl + cv::Point(i, k);
                    auto distance = sqrt((point.x - ellipse_centre.x) * (point.x - ellipse_centre.x) + (point.y - ellipse_centre.y) * (point.y - ellipse_centre.y));
                    auto alpha_angle = atan2((point.y - ellipse_centre.y), (point.x - ellipse_centre.x));
                    auto point_ellipse = ellipse_centre + cv::Point2f(ellipse_b * cos(alpha_angle + ellipse_angle), ellipse_a * sin(alpha_angle + ellipse_angle));
                    auto distance_max = sqrt((point_ellipse.x - ellipse_centre.x) * (point_ellipse.x - ellipse_centre.x) + (point_ellipse.y - ellipse_centre.y) * (point_ellipse.y - ellipse_centre.y));
                    float prob = heatMap.at<double>(point);
                    if (distance < distance_max & prob < INFINITY)
                    {
                        counter++;
                        // float prob = heatMap.at<double>(point);
                        x_keypoint += ((float)point.x) * prob;
                        y_keypoint += ((float)point.y) * prob;
                        prob_keypoint += prob;
                    }
                }
            }
            x_keypoint /= prob_keypoint;
            y_keypoint /= prob_keypoint;
            prob_keypoint /= counter;
            // cout << "prob : " << prob_keypoint << endl;
            if (x_keypoint > 0 & y_keypoint > 0)
            {
                punti.push_back(cv::Point((int)x_keypoint, (int)y_keypoint));
                keyPoints.push_back(KeyPoint_pose(cv::Point((int)x_keypoint, (int)y_keypoint), prob_keypoint));
            }

            //* plot keypoint
            //cv::circle(img_display, cv::Point(x_keypoint, y_keypoint), 5, Scalar(255, 0, 0), 3, 8);

            // cout << mu << "\n";

            // float res_rate = cutted_heat_map.rows / 2.0;
            // cv::Point mean = Point(contour_rect.x, contour_rect.y) + Point(contour_rect.height,contour_rect.width )/2.0 +Point(mu.at<double>(1, 1), mu.at<double>(2, 1)) ;
            // cout << mean << "\n";
            // cv::circle(img_display, mean, 5, Scalar(255, 0, 0), 2, 8);

            // std::cout << "covariance \n"
            //           << cov << "\n \n";

            // std::cout << "eigVec \n"
            //           << eigVec << "\n \n";

            // std::cout << "eigVal \n"
            //           << eigVal << "\n \n";
            // std::cout << "mu \n"
            //           << mu << std::endl;
            // }

            // cv::waitKey(1000);

            // cv::Mat blobMask = cv::Mat::zeros(heatMap.rows, heatMap.cols, heatMap.type());

            // cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

            // double maxVal;
            // cv::Point maxLoc;
            // cv::minMaxLoc(heatMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

            // keyPoints.push_back(KeyPoint_pose(maxLoc, heatMap.at<float>(maxLoc.y, maxLoc.x)));
        }
        // cv::imshow("test centri", img_display);
        // cv::imshow("test", UresizedHeatMap);
        // waitKey(0);

        // give a unic id number to all points
        for (int i = 0; i < keyPoints.size(); i++, keyPointId++)
        {
            keyPoints[i].id = keyPointId;
        }

        all_keyPoints->push_back(keyPoints);
        keyPointsList->insert(keyPointsList->end(), keyPoints.begin(), keyPoints.end());
    };

    //* plot for testing
    if (plot_flag)
    {
        for (int i = 0; i < ellissi.size(); i++)
        {
            cv::ellipse((*img), ellissi[i], cv::Scalar(255, 255, 0));
        }
        for (int i = 0; i < punti.size(); i++)
        {
            cv::circle((*img), punti[i], 3, cv::Scalar(255, 255, 0));
        }
    }
};

void extract_all_keypoints_cov(cv::Mat result, cv::Size targetSize,
                               vector<vector<KeyPoint_pose>> *all_keyPoints,
                               std::vector<KeyPoint_pose> *keyPointsList, cv::Mat *img, bool plot_ellipse, bool plot_circle)
{
    cv::Rect img_rect = cv::Rect(0, 0, targetSize.width, targetSize.height);
    int H = result.size[2];
    int W = result.size[3];
    int nparts = 18;
    float thresh = 0.10;
    int keyPointId = 0;
    // cv::Size blur_size = cv::Size(3, 3);
    cv::Size blur_size = cv::Size(5, 5);
    // vector<vector<KeyPoint_pose>> all_keyPoints[18];
    // find the position of the body parts
    std::vector<RotatedRect> ellissi;
    std::vector<Point> punti;
    for (int n = 0; n < nparts; n++)
    {
        vector<KeyPoint_pose> keyPoints;
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32FC1, result.ptr(0, n));
        cv::UMat UheatMap;
        heatMap.copyTo(UheatMap);
        cv::UMat UresizedHeatMap;
        cv::resize(UheatMap, UresizedHeatMap, targetSize, 0, 0, cv::INTER_NEAREST);

        cv::UMat UsmoothProbMap;
        cv::GaussianBlur(UresizedHeatMap, UsmoothProbMap, blur_size, 0, 0);
        cv::Mat smoothProbMap;
        UsmoothProbMap.copyTo(smoothProbMap);

        cv::UMat UmaskedProbMap;
        cv::threshold(UsmoothProbMap, UmaskedProbMap, thresh, 255, cv::THRESH_BINARY);
        cv::Mat maskedProbMap;
        UmaskedProbMap.copyTo(maskedProbMap);

        maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // cv::Mat img_display;
        // UresizedHeatMap.copyTo(img_display);

        cv::Mat heatmap_crop;
        for (int i = 0; i < contours.size(); i++)
        {
            // get rectangle bounding of the contour
            vector<Point> contour_poly;
            cv::approxPolyDP(contours[i], contour_poly, 3, true);
            cv::Rect contour_rect = cv::boundingRect(contours[i]);

            int tmp_w, tmp_h;
            if (!contour_rect.empty())
            {
                if (contour_rect.width > contour_rect.height)
                {
                    tmp_h = contour_rect.width;
                    tmp_w = contour_rect.width;
                }
                else
                {
                    tmp_w = contour_rect.height;
                    tmp_h = contour_rect.height;
                }
            }
            cv::Rect roi_cov = cv::Rect(contour_rect.x, contour_rect.y, contour_rect.width, contour_rect.height);
            // contour rect size n*m
            int n_max = (int)roi_cov.width;
            int m_max = (int)roi_cov.height;
            UsmoothProbMap(roi_cov).copyTo(heatmap_crop);

            uchar *data_map = heatmap_crop.data;

            // evalaute the means value for each row and col
            std::vector<float> x_means(m_max);
            std::vector<float> y_means(n_max);
            for (int m = 0; m < m_max; m++)
            {
                for (int n = 0; n < n_max; n++)
                {
                    x_means[m] += data_map[m * n_max + n] / n_max;
                    y_means[n] += data_map[m * n_max + n] / m_max;
                }
            }
            // expected value
            float x_mean = std::accumulate(x_means.begin(), x_means.end(), 0.0) / x_means.size();
            float y_mean = std::accumulate(y_means.begin(), y_means.end(), 0.0) / y_means.size();
            // square of expected value
            float x_mean_2 = x_mean * x_mean;
            float y_mean_2 = y_mean * y_mean;
            float xy_mean_2 = x_mean * y_mean;

            vector<float> xy_means;
            vector<float> xx_means;
            vector<float> yy_means;
            for (int k = 0; k < x_means.size(); k++)
            {
                xx_means.push_back(x_means[k] * x_means[k]);
                yy_means.push_back(y_means[k] * y_means[k]);
                xy_means.push_back(x_means[k] * y_means[k]);
            }
            float cov_xx = std::accumulate(xx_means.begin(), xx_means.end(), 0.0) / xx_means.size() - x_mean_2;
            float cov_yy = std::accumulate(yy_means.begin(), yy_means.end(), 0.0) / yy_means.size() - y_mean_2;
            float cov_xy = std::accumulate(xy_means.begin(), xy_means.end(), 0.0) / xy_means.size() - xy_mean_2;
            // covariance matrix
            float data[4] = {cov_xx, cov_xy, cov_xy, cov_yy};
            cv::Mat cov_matrix = cv::Mat(2, 2, CV_32F, data);
            // std::cout << "cov Matrix : " << cov_matrix << std::endl;
            // centre of ellipse
            cv::Point centre = getCentre_rect(contour_rect);
            //Calculate the error ellipse for a 95% confidence interval ( 2.4477 )
            cv::RotatedRect ellipse = getErrorEllipse(2.0, centre, cov_matrix);

            ellissi.push_back(ellipse);

            // evaluate the position of the keypoint inside the ellipse
            cv::Rect ellipse_rect = ellipse.boundingRect();
            resizeRect(&ellipse_rect, targetSize.width, targetSize.height);
            auto ellipse_a = ellipse.size.width / 2;
            auto ellipse_b = ellipse.size.height / 2;

            auto ellipse_centre = ellipse.center;
            auto ellipse_angle = ellipse.angle;
            auto tl = ellipse_rect.tl();

            float x_keypoint = 0;
            float y_keypoint = 0;
            float prob_keypoint = 0;
            int counter = 0;

            for (int i = 0; i < ellipse_rect.width; i++)
            {
                /* code */
                for (int k = 0; k < ellipse_rect.height; k++)
                {
                    /* code */
                    auto point = tl + cv::Point(i, k);
                    auto distance = sqrt((point.x - ellipse_centre.x) * (point.x - ellipse_centre.x) + (point.y - ellipse_centre.y) * (point.y - ellipse_centre.y));
                    auto alpha_angle = atan2((point.y - ellipse_centre.y), (point.x - ellipse_centre.x));
                    auto point_ellipse = ellipse_centre + cv::Point2f(ellipse_b * cos(alpha_angle + ellipse_angle), ellipse_a * sin(alpha_angle + ellipse_angle));
                    auto distance_max = sqrt((point_ellipse.x - ellipse_centre.x) * (point_ellipse.x - ellipse_centre.x) + (point_ellipse.y - ellipse_centre.y) * (point_ellipse.y - ellipse_centre.y));
                    if (img_rect.contains(point))
                    {
                        float prob = smoothProbMap.at<double>(point);
                        if (distance < distance_max & prob < INFINITY)
                        {
                            counter++;
                            // float prob = heatMap.at<double>(point);
                            x_keypoint += ((float)point.x) * prob;
                            y_keypoint += ((float)point.y) * prob;
                            prob_keypoint += prob;
                        }
                    }
                }
            }
            x_keypoint /= prob_keypoint;
            y_keypoint /= prob_keypoint;
            prob_keypoint /= counter;
            // cout << "prob : " << prob_keypoint << endl;
            if (x_keypoint > 0 & y_keypoint > 0)
            {
                punti.push_back(cv::Point((int)x_keypoint, (int)y_keypoint));
                KeyPoint_pose tmp_point = KeyPoint_pose(cv::Point((int)x_keypoint, (int)y_keypoint), prob_keypoint);
                tmp_point.ellipse = ellipse;
                keyPoints.push_back(tmp_point);
            }
        }

        // TODO: merge points inside the same ellipse ?

        // give a unic id number to all points
        for (int i = 0; i < keyPoints.size(); i++, keyPointId++)
        {
            keyPoints[i].id = keyPointId;
        }

        all_keyPoints->push_back(keyPoints);
        keyPointsList->insert(keyPointsList->end(), keyPoints.begin(), keyPoints.end());
    };

    //* plot for testing
    if (plot_ellipse)
    {
        for (int i = 0; i < ellissi.size(); i++)
        {
            try
            {
                cv::ellipse((*img), ellissi[i], cv::Scalar(255, 255, 0), 1, 8);
            }
            catch (cv::Exception &e)
            {
            }
        }
    }
    if (plot_circle)
    {
        for (int i = 0; i < punti.size(); i++)
        {
            cv::circle((*img), punti[i], 3, cv::Scalar(255, 255, 0), 1, 8);
        }
    }
    // cv::imshow("test ellissi", img);
    // int k = waitKey(10);
    // if (k == 32) // space = pause
    // {
    //     waitKey(0);
    // }
};

//_______________________________________________________________________________________
//____________ pairwise functions _______________________________________________________
void populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints, std::vector<cv::Point> &interpCoords)
{
    float xStep = ((float)(b.x - a.x)) / (float)(numPoints - 1);
    float yStep = ((float)(b.y - a.y)) / (float)(numPoints - 1);

    interpCoords.push_back(a);

    for (int i = 1; i < numPoints - 1; ++i)
    {
        interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
    }

    interpCoords.push_back(b);
}

vector<cv::Mat> getValidPairs(cv::Mat paf_blob, cv::Size img_size,
                              const std::vector<std::vector<KeyPoint_pose>> detectedKeypoints,
                              std::vector<std::vector<ValidPair>> *validPairs,
                              std::set<int> *invalidPairs)
{
    cv::Rect img_rect = cv::Rect(cv::Point(), img_size);

    // from learn opencv:
    int nInterpSamples = 10;
    float pafScoreTh = 0.1;
    float confTh = 0.70;

    // my tests
    // int nInterpSamples = 10;
    // float pafScoreTh = 0.10;
    // float confTh = 0.550;

    // int nInterpSamples = 10;
    // float pafScoreTh = 0.1;
    // float confTh = 0.45;

    // int nInterpSamples = 20;
    // float pafScoreTh = 0.1;
    // float confTh = 0.70;
    // int nInterpSamples = 100;
    // float pafScoreTh = 0.2;
    // float confTh = 0.5;

    int H = paf_blob.size[2];
    int W = paf_blob.size[3];

    cv::Mat PafA_total = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);
    cv::Mat PafB_total = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);
    cv::Mat zeros = cv::Mat::zeros(img_size.height, img_size.width, CV_32F);

    for (int k = 0; k < mapIdx.size(); ++k)
    {
        // int erosion_size = 5;
        // Mat element = getStructuringElement(cv::MORPH_RECT,
        //                                     Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        //                                     Point(erosion_size, erosion_size));
        // cout << "k  " << k << endl;
        //A->B constitute a limb
        Mat pafA(H, W, CV_32F, paf_blob.ptr(0, mapIdx[k].first - 19));
        cv::resize(pafA, pafA, img_size, 0, 0, cv::INTER_CUBIC);
        // cv::threshold(pafA, pafA, 0.01, 255, cv::THRESH_TOZERO);
        // cv::GaussianBlur(pafA, pafA, cv::Size(5, 5), 0, 0);
        // cv::dilate(pafA, pafA, element);
        // cv::medianBlur(pafA,pafA,3);
        // cv::erode(pafA, pafA, element);

        Mat pafB(H, W, CV_32F, paf_blob.ptr(0, mapIdx[k].second - 19));
        cv::resize(pafB, pafB, img_size, 0, 0, cv::INTER_CUBIC);
        // cv::threshold(pafB, pafB, 0.01, 255, cv::THRESH_TOZERO);
        // cv::GaussianBlur(pafB, pafB, cv::Size(5, 5), 0, 0);
        // cv::dilate(pafB, pafB, element);
        // cv::medianBlur(pafB, pafB,3);
        // cv::erode(pafB, pafB, element);

        //Find the keypoints for the first and second limb
        const std::vector<KeyPoint_pose> &candA = detectedKeypoints[posePairs[k].first];
        const std::vector<KeyPoint_pose> &candB = detectedKeypoints[posePairs[k].second];

        int nA = candA.size();
        int nB = candB.size();

        // imshow("PAF A", pafA);
        // imshow("PAF B", pafB);
        // waitKey(0);
        // addWeighted(pafA, 1, pafB, 1, 0.0, Paf_total);
        add(pafA * 200, zeros, PafA_total);
        add(pafB * 200, zeros, PafB_total);
        /*
		  # If keypoints for the joint-pair is detected
		  # check every joint in candA with every joint in candB
		  # Calculate the distance vector between the two joints
		  # Find the PAF values at a set of interpolated points between the joints
		  # Use the above formula to compute a score to mark the connection valid
		*/

        if (nA != 0 && nB != 0)
        {
            std::vector<ValidPair> localValidPairs;

            for (int i = 0; i < nA; ++i)
            {
                if (!img_rect.contains(candA[i].point))
                {
                    continue;
                }

                int maxJ = -1;
                float maxScore = -1;
                bool found = false;

                for (int j = 0; j < nB; ++j)
                {
                    if (!img_rect.contains(candB[j].point))
                    {
                        continue;
                    }
                    // calculate distance between cand A & cand B
                    std::pair<float, float> distance(candB[j].point.x - candA[i].point.x, candB[j].point.y - candA[i].point.y);
                    float norm = std::sqrt(distance.first * distance.first + distance.second * distance.second);

                    if (!norm)
                    {
                        continue;
                    }

                    // get unitary versor
                    distance.first /= norm;
                    distance.second /= norm;

                    //Find p(u)
                    std::vector<cv::Point> interpCoords;
                    populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);
                    //Find L(p(u))
                    std::vector<std::pair<float, float>> pafInterp;
                    for (int l = 0; l < interpCoords.size(); ++l)
                    {
                        pafInterp.push_back(
                            std::pair<float, float>(
                                pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
                                pafB.at<float>(interpCoords[l].y, interpCoords[l].x)));
                    }

                    std::vector<float> pafScores;
                    float sumOfPafScores = 0;
                    int numOverTh = 0;
                    for (int l = 0; l < pafInterp.size(); ++l)
                    {
                        float score = abs(pafInterp[l].first * distance.first + pafInterp[l].second * distance.second);
                        sumOfPafScores += score;
                        if (score > pafScoreTh)
                        {
                            ++numOverTh;
                        }

                        pafScores.push_back(score);
                    }

                    float avgPafScore = sumOfPafScores / ((float)pafInterp.size());

                    if (((float)numOverTh) / ((float)nInterpSamples) > confTh)
                    {
                        if (avgPafScore > maxScore)
                        {
                            maxJ = j;
                            maxScore = avgPafScore;
                            found = true;
                        }
                    }

                } /* j */

                if (found)
                {
                    localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
                }

            } /* i */

            validPairs->push_back(localValidPairs);
        }
        else
        {
            invalidPairs->insert(k);
            validPairs->push_back(std::vector<ValidPair>());
        }
    } /* k */
    vector<Mat> output;
    output.push_back(PafA_total);
    output.push_back(PafB_total);
    return output;
}

void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                            const std::set<int> &invalidPairs,
                            std::vector<std::vector<int>> &personwiseKeypoints)
{
    for (int k = 0; k < mapIdx.size(); ++k)
    {
        if (invalidPairs.find(k) != invalidPairs.end())
        {
            continue;
        }

        const std::vector<ValidPair> &localValidPairs(validPairs[k]);

        int indexA(posePairs[k].first);
        int indexB(posePairs[k].second);

        for (int i = 0; i < localValidPairs.size(); ++i)
        {
            bool found = false;
            int personIdx = -1;

            for (int j = 0; !found && j < personwiseKeypoints.size(); ++j)
            {
                if (indexA < personwiseKeypoints[j].size() &&
                    personwiseKeypoints[j][indexA] == localValidPairs[i].aId)
                {
                    personIdx = j;
                    found = true;
                }
            } /* j */

            if (found)
            {
                personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
            }
            else if (k < 17)
            {
                std::vector<int> lpkp(std::vector<int>(18, -1));

                lpkp.at(indexA) = localValidPairs[i].aId;
                lpkp.at(indexB) = localValidPairs[i].bId;

                personwiseKeypoints.push_back(lpkp);
            }

        } /* i */
    }     /* k */
}

void plot_all_skeleton(cv::Mat *img, std::vector<std::vector<int>> personwiseKeypoints,
                       std::vector<KeyPoint_pose> keyPointsList, bool white)
{

    for (int i = 0; i < 17; i++)
    {
        for (int n = 0; n < personwiseKeypoints.size(); ++n)
        {
            const std::pair<int, int> &posePair = posePairs[i];
            int indexA = personwiseKeypoints[n][posePair.first];
            int indexB = personwiseKeypoints[n][posePair.second];

            if (indexA == -1 || indexB == -1)
            {
                continue;
            }

            KeyPoint_pose &kpA = keyPointsList[indexA];
            KeyPoint_pose &kpB = keyPointsList[indexB];

            if (white)
            {
                cv::line((*img), kpA.point, kpB.point, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                try
                {
                    cv::ellipse((*img), kpA.ellipse, cv::Scalar(0, 0, 255), 1, 8);
                }
                catch (const std::exception &e)
                {
                    // std::cerr << e.what() << '\n';
                }
                try
                {
                    cv::ellipse((*img), kpB.ellipse, cv::Scalar(0, 0, 255), 1, 8);
                }
                catch (const std::exception &e)
                {
                    // std::cerr << e.what() << '\n';
                }
            }
            else
            {
                cv::line((*img), kpA.point, kpB.point, colors_left_right[i], 2, cv::LINE_AA);
            }
        }
    }
}

void plot_skeleton(cv::Mat *img, std::vector<int> personwiseKeypoints,
                   std::vector<KeyPoint_pose> keyPointsList, bool white)
{

    for (int i = 0; i < 17; i++)
    {

        const std::pair<int, int> &posePair = posePairs[i];
        int indexA = personwiseKeypoints[posePair.first];
        int indexB = personwiseKeypoints[posePair.second];

        if (indexA == -1 || indexB == -1)
        {
            continue;
        }

        KeyPoint_pose &kpA = keyPointsList[indexA];
        KeyPoint_pose &kpB = keyPointsList[indexB];

        if (white)
        {
            cv::line((*img), kpA.point, kpB.point, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
        }
        else
        {
            cv::line((*img), kpA.point, kpB.point, colors_left_right[i], 3, cv::LINE_AA);
        }
    }
}

std::vector<std::vector<int>> good_skeleton_extraction(std::vector<std::vector<int>> personwiseKeypoints, int min_pairs_th)
{
    std::vector<std::vector<int>> good_skeleton;
    int counter_pairs = 0;
    for (int n = 0; n < personwiseKeypoints.size(); ++n)
    {
        for (int i = 0; i < 17; i++)
        {
            const std::pair<int, int> &posePair = posePairs[i];
            int indexA = personwiseKeypoints[n][posePair.first];
            int indexB = personwiseKeypoints[n][posePair.second];

            if (indexA == -1 || indexB == -1)
            {
                continue;
            }
            else
            {
                counter_pairs++;
            }
        }
        if (counter_pairs > min_pairs_th)
        {
            good_skeleton.push_back(personwiseKeypoints[n]);
            counter_pairs = 0;
        }
        else
        {
            counter_pairs = 0;
        }
    }
    return good_skeleton;
}

//// functions in test

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