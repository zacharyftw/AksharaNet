#pragma once

#include <string>

namespace aksharanet {

class LanguageModel {
public:
    explicit LanguageModel(const std::string& lm_path);
    std::string correct(const std::string& text);
};

} // namespace aksharanet
