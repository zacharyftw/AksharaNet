#include <gtest/gtest.h>
#include <aksharanet/decoding/decoder.hpp>

TEST(Decoding, DecodeEmptyLogitsReturnsEmptyString) {
    aksharanet::Decoder d;
    std::string result = d.decode({});
    EXPECT_TRUE(result.empty());
}
