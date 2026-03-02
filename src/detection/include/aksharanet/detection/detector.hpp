#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace aksharanet {

struct TextRegion {
    cv::Rect bounding_box;
    float confidence;
};

class Detector {
public:
    std::vector<TextRegion> detect(const cv::Mat& image);
};

} // namespace aksharanet
