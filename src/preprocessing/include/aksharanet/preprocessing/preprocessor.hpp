#pragma once

#include <opencv2/core.hpp>

namespace aksharanet {

class Preprocessor {
public:
    // Full pipeline: denoise → binarize → deskew → (future stages)
    cv::Mat process(const cv::Mat& input);

    // Sauvola adaptive binarization — handles uneven lighting and thin Malayalam strokes
    cv::Mat binarize(const cv::Mat& gray);

    // Gaussian denoising before binarization
    cv::Mat denoise(const cv::Mat& gray);

    // Detect skew angle (degrees) via Hough line transform
    double detect_skew(const cv::Mat& binary);

    // Rotate image to correct detected skew
    cv::Mat deskew(const cv::Mat& binary);

    // CLAHE contrast normalization — boosts faded/low-contrast scans
    cv::Mat normalize_contrast(const cv::Mat& gray);

    // Perspective correction — flattens images taken at an angle
    cv::Mat correct_perspective(const cv::Mat& gray);
};

} // namespace aksharanet
