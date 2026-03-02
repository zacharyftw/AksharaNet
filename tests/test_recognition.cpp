#include <gtest/gtest.h>
#include <aksharanet/recognition/recognizer.hpp>
#include <opencv2/core.hpp>

TEST(Recognition, RecognizeReturnsEmptyLogitsForStub) {
    aksharanet::Recognizer r("");
    cv::Mat patch = cv::Mat::zeros(32, 128, CV_8UC3);
    auto logits = r.recognize(patch);
    EXPECT_TRUE(logits.empty());
}
