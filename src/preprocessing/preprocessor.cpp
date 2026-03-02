#include <aksharanet/preprocessing/preprocessor.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace aksharanet {

cv::Mat Preprocessor::denoise(const cv::Mat& gray) {
    cv::Mat result;
    // Gaussian blur — mild kernel to remove scan grain without killing thin strokes
    cv::GaussianBlur(gray, result, cv::Size(3, 3), 0);
    return result;
}

cv::Mat Preprocessor::binarize(const cv::Mat& gray) {
    // Sauvola adaptive binarization:
    // For each pixel, threshold = mean * (1 + k * (std / R - 1))
    // k=0.2, R=128 are standard defaults. Window size 51 works well for Malayalam print.
    const int    window_size = 51;
    const double k           = 0.2;
    const double R           = 128.0;

    cv::Mat mean, mean_sq, std_dev;

    // Compute local mean and std using integral images
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_64F);

    cv::boxFilter(gray_f, mean,    CV_64F, cv::Size(window_size, window_size));
    cv::boxFilter(gray_f.mul(gray_f), mean_sq, CV_64F, cv::Size(window_size, window_size));

    cv::sqrt(cv::max(mean_sq - mean.mul(mean), 0.0), std_dev);

    cv::Mat threshold = mean.mul(1.0 + k * (std_dev / R - 1.0));

    cv::Mat binary(gray.size(), CV_8U);
    for (int r = 0; r < gray.rows; ++r) {
        const uchar*  src = gray.ptr<uchar>(r);
        const double* thr = threshold.ptr<double>(r);
        uchar*        dst = binary.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            dst[c] = (static_cast<double>(src[c]) < thr[c]) ? 0 : 255;
    }

    return binary;
}

double Preprocessor::detect_skew(const cv::Mat& binary) {
    // Invert so text is white on black, required for HoughLinesP
    cv::Mat inverted;
    cv::bitwise_not(binary, inverted);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(inverted, lines, 1, CV_PI / 180.0, 80, 30, 10);

    if (lines.empty())
        return 0.0;

    double angle_sum = 0.0;
    int    count     = 0;
    for (const auto& l : lines) {
        double angle = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        // Only consider near-horizontal lines (text baselines)
        if (std::abs(angle) < 45.0) {
            angle_sum += angle;
            ++count;
        }
    }

    return count > 0 ? angle_sum / count : 0.0;
}

cv::Mat Preprocessor::deskew(const cv::Mat& binary) {
    double angle = detect_skew(binary);

    // Skip rotation if skew is negligible
    if (std::abs(angle) < 0.1)
        return binary.clone();

    cv::Point2f center(binary.cols / 2.0f, binary.rows / 2.0f);
    cv::Mat     rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat     result;
    cv::warpAffine(binary, result, rot, binary.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
    return result;
}

cv::Mat Preprocessor::process(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    gray = denoise(gray);
    gray = binarize(gray);
    gray = deskew(gray);
    return gray;
}

} // namespace aksharanet
