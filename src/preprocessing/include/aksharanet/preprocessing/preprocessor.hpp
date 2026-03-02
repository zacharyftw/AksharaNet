#pragma once

#include <opencv2/core.hpp>

namespace aksharanet {

class Preprocessor {
public:
    // Full pipeline: denoise → binarize → (future stages)
    cv::Mat process(const cv::Mat& input);

    // Sauvola adaptive binarization — handles uneven lighting and thin Malayalam strokes
    cv::Mat binarize(const cv::Mat& gray);

    // Gaussian denoising before binarization
    cv::Mat denoise(const cv::Mat& gray);
};

} // namespace aksharanet
