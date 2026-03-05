#pragma once

#include <string>

namespace aksharanet {

class LanguageModel {
public:
    // Load KenLM model from .arpa or binary file
    // Empty path = no LM scoring (passthrough)
    explicit LanguageModel(const std::string& lm_path);

    // Score a character sequence using the loaded n-gram model
    // Returns log10 probability
    float score(const std::string& text) const;

    // Full post-processing: NFC normalize → chillu normalize → ZWJ strip
    std::string correct(const std::string& text) const;

    // NFC Unicode normalization via ICU4C
    static std::string normalize_nfc(const std::string& text);

    // Normalize chillu forms to canonical representation
    static std::string normalize_chillu(const std::string& text);

    // Strip spurious ZWJ/ZWNJ that don't participate in valid conjuncts
    static std::string strip_zwj(const std::string& text);

    bool has_model() const { return has_model_; }

private:
    bool has_model_ = false;
};

} // namespace aksharanet
