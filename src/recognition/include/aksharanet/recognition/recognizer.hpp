#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace aksharanet {

class Recognizer {
public:
    explicit Recognizer(const std::string& model_path);
    std::vector<float> recognize(const cv::Mat& patch);
};

} // namespace aksharanet
