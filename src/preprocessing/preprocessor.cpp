#include <aksharanet/preprocessing/preprocessor.hpp>

namespace aksharanet {

cv::Mat Preprocessor::process(const cv::Mat& input) {
    return input.clone();
}

} // namespace aksharanet
