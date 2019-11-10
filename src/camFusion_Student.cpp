#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(
    std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints,
    float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


//associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(
    BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev,
    std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches)
{
    std::vector<std::tuple<double, cv::KeyPoint, cv::DMatch>> tmpResults;

    double mean = 0.0;
    double variance = 0.0;
    int numSamples = 0;
    for (const auto& kptMatch : kptMatches)
    {
        auto ptCurr = kptsCurr.at(kptMatch.trainIdx);
        if (boundingBox.roi.contains(ptCurr.pt))
        {
            auto ptPrev = kptsPrev.at(kptMatch.queryIdx);
            double distance = cv::norm(ptCurr.pt - ptPrev.pt);
            ++numSamples;
            double meanUpdated = mean + (distance - mean) / static_cast<double>(numSamples);
            variance += ((distance - mean) * (distance - meanUpdated) - variance) / numSamples;
            mean = meanUpdated;

            tmpResults.push_back(std::make_tuple(distance, ptCurr, kptMatch));
        }
    }
    double stdDev = std::sqrt(variance);

    for (const auto& tmpTuple : tmpResults)
    {
        if (std::get<0>(tmpTuple) < (mean + 0.8 * stdDev))
        {
            boundingBox.keypoints.push_back(std::get<1>(tmpTuple));
            boundingBox.kptMatches.push_back(std::get<2>(tmpTuple));
        }
    }
}

// get the median value of the input vector's elements (input function determines which members are used)
template<class T>
double median(
    const std::vector<T>& input, const std::function<double(const T& e1)>& extractVal)
{
    if (input.empty())
    {
        throw std::runtime_error("Can't compute median for empty vector");
    }
    std::vector<T> inputCopy(input.begin(), input.end());

    const size_t n = inputCopy.size() / 2;
    auto compFunction =
        [&extractVal](const T& e1, const T& e2) {
            return extractVal(e1) < extractVal(e2);
        };
    std::nth_element(inputCopy.begin(), inputCopy.begin() + n, inputCopy.end(), compFunction);
    auto elemN = inputCopy[n];
    if (inputCopy.size() % 2 == 1)
    {
        return extractVal(elemN);
    }
    else
    {
        std::nth_element(inputCopy.begin(), inputCopy.begin() + n - 1, inputCopy.end(), compFunction);
        return 0.5 * (extractVal(elemN) + extractVal(inputCopy[n - 1]));
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(
    std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr,
    std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, cv::Mat* visImg)
{
    vector<double> distRatios;
    if (!kptMatches.empty())
    {
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
        {
            cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
            cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

            for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
            {
                double minDist = 100.0;
                cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
                cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

                double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
                double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

                if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
                {
                    double distRatio = distCurr / distPrev;
                    distRatios.push_back(distRatio);
                }
            }
        }
    }

    if (distRatios.empty())
    {
        TTC = NAN;
        return;
    }

    std::function<double(const double& e)> extractVal = [](const double& e) {return e;};
    double medDistRatio = median(distRatios, extractVal);
    if (std::fabs(1 - medDistRatio) < std::numeric_limits<double>::epsilon())
    {
        TTC = NAN;
        return;
    }
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(
    std::vector<LidarPoint>& lidarPointsPrev,
    std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC)
{
    if (std::fabs(frameRate) < std::numeric_limits<double>::epsilon())
    {
        throw std::runtime_error("Frame rate must be greater than 0.0");
    }

    std::function<double(const LidarPoint&)> extractX = [](const LidarPoint& e1) {return e1.x;};
    double prevMedianX = median(lidarPointsPrev, extractX);
    double currMedianX = median(lidarPointsCurr, extractX);
    double dt = 1 / frameRate;
    double medianDiff = prevMedianX - currMedianX;
    if (std::fabs(medianDiff) < std::numeric_limits<double>::epsilon())
    {
        TTC = NAN;
        return;
    }
    TTC = currMedianX * dt / (medianDiff);
}


void matchBoundingBoxes(
    std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches, DataFrame& prevFrame,
    DataFrame& currFrame)
{
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, unsigned int>> bbMatches;
    for (const auto& kpMatch : matches)
    {
        auto queryPoint = prevFrame.keypoints.at(static_cast<size_t>(kpMatch.queryIdx)).pt;
        auto trainPoint = currFrame.keypoints.at(static_cast<size_t>(kpMatch.trainIdx)).pt;
        for (unsigned int i = 0; i < prevFrame.boundingBoxes.size(); ++i)
        {
            if (prevFrame.boundingBoxes.at(i).roi.contains(queryPoint))
            {
                for (unsigned int j = 0; j < currFrame.boundingBoxes.size(); ++j)
                {
                    if (currFrame.boundingBoxes.at(j).roi.contains(trainPoint))
                    {
                        bbMatches[i][j] += 1;
                    }
                }
            }
        }
    }

    for (const auto& bbMatch : bbMatches)
    {
        auto maxElement = std::max_element(bbMatch.second.begin(), bbMatch.second.end(),
                [](const std::pair<unsigned int, unsigned int>& e1,
                const std::pair<unsigned int, unsigned int>& e2) {
                    return e1.second < e2.second;
                });

        if (maxElement != bbMatch.second.end() && maxElement->second > 0)
        {
            bbBestMatches.insert(std::make_pair(prevFrame.boundingBoxes.at(bbMatch.first).boxID,
                currFrame.boundingBoxes.at(maxElement->first).boxID));
        }
    }
}
