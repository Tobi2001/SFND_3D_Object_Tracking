#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(
    std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
    std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        if (descriptorType == "DES_HOG")
        {
            normType = cv::NORM_L2;
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {

        if (descriptorType == "DES_HOG")
        {
            matcher = cv::FlannBasedMatcher::create();
        }
        else
        {
            matcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));
        }
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        const float ratio_thresh = 0.8f;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, string descriptorType)
{
    //BRIEF, ORB, FREAK, AKAZE, SIFT
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        // BRISK is the default value
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
        descriptorType = "BRISK";
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

void visualizeKeypoints(const cv::Mat& img, const vector<cv::KeyPoint>& keypoints, const std::string& windowName)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results");
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis)
{
    int blockSize = 2;
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;

    double t = static_cast<double>(cv::getTickCount());
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    float maxOverlap = 0.0;
    for (int j = 0; j < dst_norm.rows; j++)
    {
        for (int i = 0; i < dst_norm.cols; i++)
        {
            int response = static_cast<int>(dst_norm.at<float>(j, i));
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    float kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, "Harris Detector Results");
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis)
{
    double t = static_cast<double>(cv::getTickCount());

    if (detectorType == "BRISK")
    {
        auto briskDetector = cv::BRISK::create();
        briskDetector->detect(img, keypoints);
    }
    else if (detectorType == "ORB")
    {
        auto orbDetector = cv::ORB::create();
        orbDetector->detect(img, keypoints);
    }
    else if (detectorType == "AKAZE")
    {
        auto akazeDetector = cv::AKAZE::create();
        akazeDetector->detect(img, keypoints);
    }
    else if (detectorType == "SIFT")
    {
        auto siftDetector = cv::xfeatures2d::SIFT::create();
        siftDetector->detect(img, keypoints);
    }
    else
    {
        // FAST is default value
        cv::FAST(img, keypoints, 100);
        detectorType = "FAST";
    }

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" <<
        endl;

    // visualize results
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, detectorType + " Detector Results");
    }
}
