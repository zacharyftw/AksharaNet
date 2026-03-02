#include <aksharanet/preprocessing/preprocessor.hpp>
#include <opencv2/imgproc.hpp>

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

cv::Mat Preprocessor::process(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    gray = denoise(gray);
    gray = binarize(gray);
    return gray;
}

} // namespace aksharanet
