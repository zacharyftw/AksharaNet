#pragma once

#include <opencv2/core.hpp>
#include <vector>

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

    // Resize to target dimensions with padding to preserve aspect ratio
    // Pads with white (255) to avoid stretching Malayalam vowel signs
    cv::Mat resize_with_padding(const cv::Mat& gray, int target_w, int target_h);

    // Returns true if the image is a clean digital render (PDF screenshot etc.)
    // Clean renders skip binarization to avoid degrading already-perfect pixels
    bool is_clean_render(const cv::Mat& gray);

    // Segment a binarized image into individual text line crops
    // Uses horizontal projection profile — finds valleys between text rows
    std::vector<cv::Mat> segment_lines(const cv::Mat& binary);
};

} // namespace aksharanet
