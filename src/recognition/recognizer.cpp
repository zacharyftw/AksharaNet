#include <aksharanet/recognition/recognizer.hpp>

namespace aksharanet {

Recognizer::Recognizer(const std::string& /*model_path*/) {}

std::vector<float> Recognizer::recognize(const cv::Mat& /*patch*/) {
    return {};
}

} // namespace aksharanet
