#include <aksharanet/language_model/language_model.hpp>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>
#include <string>
#include <vector>

namespace aksharanet {

// Unicode constants
static constexpr char32_t ZWJ  = 0x200D;
static constexpr char32_t ZWNJ = 0x200C;
static constexpr char32_t VIRAMA = 0x0D4D; // Malayalam virama (chandrakkala)

// Chillu codepoints and their virama-based equivalents
// Chillu form → base consonant + virama + ZWJ
struct ChilluMapping {
    char32_t chillu;
    char32_t base;
};

static constexpr ChilluMapping chillu_map[] = {
    {0x0D7A, 0x0D23}, // ണ്‍ → ണ
    {0x0D7B, 0x0D28}, // ന്‍ → ന
    {0x0D7C, 0x0D30}, // ര്‍ → ര
    {0x0D7D, 0x0D32}, // ല്‍ → ല
    {0x0D7E, 0x0D33}, // ള്‍ → ള
    {0x0D7F, 0x0D15}, // ക്‍ → ക
};

// Helper: encode a char32_t to UTF-8
static void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Helper: decode UTF-8 to codepoints
static std::vector<char32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<char32_t> result;
    size_t i = 0;
    while (i < s.size()) {
        char32_t cp = 0;
        auto b = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (b < 0x80) {
            cp = b;
        } else if ((b & 0xE0) == 0xC0) {
            cp = b & 0x1F; len = 2;
        } else if ((b & 0xF0) == 0xE0) {
            cp = b & 0x0F; len = 3;
        } else if ((b & 0xF8) == 0xF0) {
            cp = b & 0x07; len = 4;
        }
        for (int j = 1; j < len && i + j < s.size(); ++j)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        result.push_back(cp);
        i += len;
    }
    return result;
}

LanguageModel::LanguageModel(const std::string& lm_path) {
    // TODO: load KenLM model when .arpa/.bin file is provided
    has_model_ = !lm_path.empty();
}

float LanguageModel::score(const std::string& /*text*/) const {
    // TODO: query KenLM model for n-gram score
    // Returns 0 (log10(1) = neutral) when no model is loaded
    return 0.0f;
}

std::string LanguageModel::normalize_nfc(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status))
        return text;

    icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(text);
    icu::UnicodeString normalized = normalizer->normalize(ustr, status);
    if (U_FAILURE(status))
        return text;

    std::string result;
    normalized.toUTF8String(result);
    return result;
}

std::string LanguageModel::normalize_chillu(const std::string& text) {
    auto cps = utf8_to_codepoints(text);
    std::string result;

    for (size_t i = 0; i < cps.size(); ++i) {
        bool replaced = false;

        // Check for base + virama + ZWJ sequences → replace with chillu
        if (i + 2 < cps.size() && cps[i + 1] == VIRAMA && cps[i + 2] == ZWJ) {
            for (const auto& m : chillu_map) {
                if (cps[i] == m.base) {
                    append_utf8(result, m.chillu);
                    i += 2; // skip virama + ZWJ
                    replaced = true;
                    break;
                }
            }
        }

        if (!replaced)
            append_utf8(result, cps[i]);
    }

    return result;
}

std::string LanguageModel::strip_zwj(const std::string& text) {
    auto cps = utf8_to_codepoints(text);
    std::string result;

    for (size_t i = 0; i < cps.size(); ++i) {
        if (cps[i] == ZWJ || cps[i] == ZWNJ) {
            // Keep ZWJ/ZWNJ only between a virama and a consonant (valid conjunct)
            bool after_virama = (i > 0 && cps[i - 1] == VIRAMA);
            bool before_consonant = (i + 1 < cps.size() &&
                                     cps[i + 1] >= 0x0D15 && cps[i + 1] <= 0x0D39);
            if (after_virama && before_consonant) {
                append_utf8(result, cps[i]);
                continue;
            }
            // Otherwise strip it
            continue;
        }
        append_utf8(result, cps[i]);
    }

    return result;
}

std::string LanguageModel::correct(const std::string& text) const {
    std::string result = normalize_nfc(text);
    result = normalize_chillu(result);
    result = strip_zwj(result);
    return result;
}

} // namespace aksharanet
