#include <gtest/gtest.h>
#include <aksharanet/language_model/language_model.hpp>

TEST(LanguageModel, CorrectPassesThrough) {
    aksharanet::LanguageModel lm("");
    std::string result = lm.correct("hello");
    EXPECT_EQ(result, "hello");
}

TEST(LanguageModel, ScoreReturnsZeroWithoutModel) {
    aksharanet::LanguageModel lm("");
    EXPECT_FLOAT_EQ(lm.score("test"), 0.0f);
    EXPECT_FALSE(lm.has_model());
}

TEST(LanguageModel, NfcNormalizesDecomposed) {
    // NFC should compose decomposed sequences
    // e.g. 'é' as e + combining acute → single codepoint
    std::string decomposed = "e\xCC\x81"; // e + combining acute accent
    std::string composed = "\xC3\xA9";     // é
    EXPECT_EQ(aksharanet::LanguageModel::normalize_nfc(decomposed), composed);
}

TEST(LanguageModel, NfcPassesThroughAlreadyComposed) {
    std::string text = "hello world";
    EXPECT_EQ(aksharanet::LanguageModel::normalize_nfc(text), text);
}

TEST(LanguageModel, ChilluNormalizesViramaZwjSequence) {
    // ന (0D28) + ് (0D4D virama) + ZWJ (200D) → ൻ (0D7B chillu)
    std::string virama_form = "\xE0\xB4\xA8\xE0\xB5\x8D\xE2\x80\x8D";
    std::string chillu_form = "\xE0\xB5\xBB";
    EXPECT_EQ(aksharanet::LanguageModel::normalize_chillu(virama_form), chillu_form);
}

TEST(LanguageModel, StripZwjRemovesStrayZwj) {
    // Standalone ZWJ not between virama+consonant should be stripped
    std::string with_stray = std::string("a") + "\xE2\x80\x8D" + "b"; // a + ZWJ + b
    EXPECT_EQ(aksharanet::LanguageModel::strip_zwj(with_stray), "ab");
}

TEST(LanguageModel, StripZwjKeepsValidConjunct) {
    // virama (0D4D) + ZWJ (200D) + consonant (0D15 ക) should be kept
    std::string valid = "\xE0\xB5\x8D\xE2\x80\x8D\xE0\xB4\x95";
    EXPECT_EQ(aksharanet::LanguageModel::strip_zwj(valid), valid);
}
