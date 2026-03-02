#include <gtest/gtest.h>
#include <aksharanet/preprocessing/preprocessor.hpp>
#include <opencv2/core.hpp>

TEST(Preprocessing, ProcessReturnsNonEmptyMat) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::zeros(64, 256, CV_8UC3);
    cv::Mat output = p.process(input);
    EXPECT_FALSE(output.empty());
}

TEST(Preprocessing, OutputIsSingleChannel) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(64, 256, CV_8UC3) * 128;
    cv::Mat output = p.process(input);
    EXPECT_EQ(output.channels(), 1);
}

TEST(Preprocessing, DenoisePreservesSize) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(100, 200, CV_8UC1) * 200;
    cv::Mat output = p.denoise(input);
    EXPECT_EQ(output.size(), input.size());
}

TEST(Preprocessing, BinarizeOutputIsBinary) {
    aksharanet::Preprocessor p;
    // Uniform grey image — all pixels should binarize to 255 (background)
    cv::Mat input = cv::Mat::ones(100, 100, CV_8UC1) * 200;
    cv::Mat output = p.binarize(input);
    double minVal, maxVal;
    cv::minMaxLoc(output, &minVal, &maxVal);
    EXPECT_TRUE(minVal == 0.0 || minVal == 255.0);
    EXPECT_TRUE(maxVal == 0.0 || maxVal == 255.0);
}
