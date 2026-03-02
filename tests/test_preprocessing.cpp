#include <gtest/gtest.h>
#include <aksharanet/preprocessing/preprocessor.hpp>
#include <opencv2/core.hpp>

TEST(Preprocessing, ProcessReturnsNonEmptyMat) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::zeros(64, 256, CV_8UC3);
    cv::Mat output = p.process(input);
    EXPECT_FALSE(output.empty());
}
