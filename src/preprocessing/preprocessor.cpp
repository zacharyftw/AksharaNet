#include <aksharanet/preprocessing/preprocessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
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

cv::Mat Preprocessor::normalize_contrast(const cv::Mat& gray) {
    // CLAHE — adaptive histogram equalization with clip limit to avoid
    // amplifying noise in flat regions while boosting faded text
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat result;
    clahe->apply(gray, result);
    return result;
}

cv::Mat Preprocessor::correct_perspective(const cv::Mat& gray) {
    // Detect document edges using Canny + contour finding
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return gray.clone();

    // Find the largest contour — assumed to be the document boundary
    auto largest = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    std::vector<cv::Point> approx;
    cv::approxPolyDP(*largest, approx, cv::arcLength(*largest, true) * 0.02, true);

    // Only correct if we found a quadrilateral
    if (approx.size() != 4)
        return gray.clone();

    // Sort corners: top-left, top-right, bottom-right, bottom-left
    std::sort(approx.begin(), approx.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    });

    std::vector<cv::Point2f> src = {
        cv::Point2f(approx[0]),
        cv::Point2f(approx[1]),
        cv::Point2f(approx[3]),
        cv::Point2f(approx[2])
    };

    float w = static_cast<float>(gray.cols);
    float h = static_cast<float>(gray.rows);
    std::vector<cv::Point2f> dst = {
        {0, 0}, {w, 0}, {0, h}, {w, h}
    };

    cv::Mat transform = cv::getPerspectiveTransform(src, dst);
    cv::Mat result;
    cv::warpPerspective(gray, result, transform, gray.size(),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
    return result;
}

cv::Mat Preprocessor::resize_with_padding(const cv::Mat& gray, int target_w, int target_h) {
    float scale = std::min(
        static_cast<float>(target_w) / gray.cols,
        static_cast<float>(target_h) / gray.rows
    );

    int scaled_w = static_cast<int>(gray.cols * scale);
    int scaled_h = static_cast<int>(gray.rows * scale);

    cv::Mat scaled;
    cv::resize(gray, scaled, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);

    // Place scaled image top-left, pad remainder with white
    cv::Mat canvas(target_h, target_w, CV_8UC1, cv::Scalar(255));
    scaled.copyTo(canvas(cv::Rect(0, 0, scaled_w, scaled_h)));
    return canvas;
}

bool Preprocessor::is_clean_render(const cv::Mat& gray) {
    // Clean digital renders have very few unique grey values —
    // pixels are either near-black (text) or near-white (background)
    // with almost nothing in between. Scanned images have broad histograms.
    cv::Mat hist;
    const int    channels[] = {0};
    const int    hist_size  = 256;
    const float  range[]    = {0, 256};
    const float* ranges[]   = {range};

    cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, &hist_size, ranges);

    // Count bins in the mid-grey range (64–192) that have significant mass
    float total  = static_cast<float>(gray.total());
    float midsum = 0.0f;
    for (int i = 64; i < 192; ++i)
        midsum += hist.at<float>(i);

    // If less than 2% of pixels are mid-grey, it's likely a clean render
    return (midsum / total) < 0.02f;
}

std::vector<cv::Mat> Preprocessor::segment_lines(const cv::Mat& binary) {
    // Horizontal projection: count black (text) pixels per row
    // Binary image convention: 0 = text, 255 = background
    std::vector<int> projection(binary.rows, 0);
    for (int r = 0; r < binary.rows; ++r)
        for (int c = 0; c < binary.cols; ++c)
            if (binary.at<uchar>(r, c) == 0)
                ++projection[r];

    // Find text line bands: rows where projection > threshold
    const int min_text_pixels = std::max(1, binary.cols / 50);
    const int min_line_height = 5;

    std::vector<cv::Mat> lines;
    int line_start = -1;

    for (int r = 0; r <= binary.rows; ++r) {
        bool is_text_row = (r < binary.rows) && (projection[r] >= min_text_pixels);

        if (is_text_row && line_start < 0) {
            line_start = r;
        } else if (!is_text_row && line_start >= 0) {
            int height = r - line_start;
            if (height >= min_line_height)
                lines.push_back(binary(cv::Rect(0, line_start, binary.cols, height)).clone());
            line_start = -1;
        }
    }

    return lines;
}

cv::Mat Preprocessor::process(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    if (is_clean_render(gray)) {
        // Clean digital render — skip denoising and binarization,
        // they would only degrade already-perfect pixels
        gray = deskew(gray);
    } else {
        gray = normalize_contrast(gray);
        gray = denoise(gray);
        gray = binarize(gray);
        gray = deskew(gray);
    }
    return gray;
}

} // namespace aksharanet
