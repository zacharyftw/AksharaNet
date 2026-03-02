#pragma once

#include <opencv2/core.hpp>

namespace aksharanet {

class Preprocessor {
public:
    cv::Mat process(const cv::Mat& input);
};

} // namespace aksharanet
