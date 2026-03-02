#include <gtest/gtest.h>
#include <aksharanet/language_model/language_model.hpp>

TEST(LanguageModel, CorrectPassesThroughInStub) {
    aksharanet::LanguageModel lm("");
    std::string result = lm.correct("test input");
    EXPECT_EQ(result, "test input");
}
