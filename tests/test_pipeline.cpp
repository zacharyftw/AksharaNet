#include <gtest/gtest.h>
#include <aksharanet/pipeline/pipeline.hpp>

TEST(Pipeline, RunReturnsEmptyStringInStub) {
    aksharanet::PipelineConfig config;
    aksharanet::Pipeline p(config);
    std::string result = p.run("");
    EXPECT_TRUE(result.empty());
}
