#include <gtest/gtest.h>
#include <aksharanet/detection/detector.hpp>
#include <opencv2/core.hpp>

TEST(Detection, DetectReturnsEmptyForBlankImage) {
    aksharanet::Detector d;
    cv::Mat image = cv::Mat::zeros(256, 256, CV_8UC3);
    auto regions = d.detect(image);
    EXPECT_TRUE(regions.empty());
}
