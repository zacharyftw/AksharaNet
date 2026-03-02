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

TEST(Preprocessing, DeskewPreservesSize) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(200, 400, CV_8UC1) * 255;
    cv::Mat output = p.deskew(input);
    EXPECT_EQ(output.size(), input.size());
}

TEST(Preprocessing, DetectSkewOnBlankIsZero) {
    aksharanet::Preprocessor p;
    // Blank white image has no lines — skew should be 0
    cv::Mat input = cv::Mat::ones(200, 400, CV_8UC1) * 255;
    double angle = p.detect_skew(input);
    EXPECT_NEAR(angle, 0.0, 0.01);
}

TEST(Preprocessing, NormalizeContrastPreservesSizeAndType) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(100, 200, CV_8UC1) * 100;
    cv::Mat output = p.normalize_contrast(input);
    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(output.type(), input.type());
}

TEST(Preprocessing, CorrectPerspectivePreservesSize) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(200, 400, CV_8UC1) * 255;
    cv::Mat output = p.correct_perspective(input);
    EXPECT_EQ(output.size(), input.size());
}

TEST(Preprocessing, ResizeWithPaddingProducesTargetSize) {
    aksharanet::Preprocessor p;
    cv::Mat input = cv::Mat::ones(64, 512, CV_8UC1) * 128;
    cv::Mat output = p.resize_with_padding(input, 128, 32);
    EXPECT_EQ(output.cols, 128);
    EXPECT_EQ(output.rows, 32);
}

TEST(Preprocessing, IsCleanRenderDetectsHighContrastImage) {
    aksharanet::Preprocessor p;
    // Pure white image — all pixels in background bin, nothing mid-grey
    cv::Mat clean = cv::Mat::ones(100, 100, CV_8UC1) * 255;
    EXPECT_TRUE(p.is_clean_render(clean));
}

TEST(Preprocessing, IsCleanRenderRejectsMidGreyImage) {
    aksharanet::Preprocessor p;
    // Uniform mid-grey — typical of scanned/noisy images
    cv::Mat scan = cv::Mat::ones(100, 100, CV_8UC1) * 128;
    EXPECT_FALSE(p.is_clean_render(scan));
}

TEST(Preprocessing, SegmentLinesOnBlankReturnsEmpty) {
    aksharanet::Preprocessor p;
    // All-white image has no text rows — should return no lines
    cv::Mat blank = cv::Mat::ones(200, 400, CV_8UC1) * 255;
    auto lines = p.segment_lines(blank);
    EXPECT_TRUE(lines.empty());
}

TEST(Preprocessing, SegmentLinesFindsTextBands) {
    aksharanet::Preprocessor p;
    // Create image with two horizontal black bands (simulated text lines)
    cv::Mat img = cv::Mat::ones(100, 200, CV_8UC1) * 255;
    img(cv::Rect(0, 10, 200, 15)).setTo(0); // line 1
    img(cv::Rect(0, 50, 200, 15)).setTo(0); // line 2
    auto lines = p.segment_lines(img);
    EXPECT_EQ(static_cast<int>(lines.size()), 2);
}
