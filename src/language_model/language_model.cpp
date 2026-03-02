#include <aksharanet/language_model/language_model.hpp>

namespace aksharanet {

LanguageModel::LanguageModel(const std::string& /*lm_path*/) {}

std::string LanguageModel::correct(const std::string& text) {
    return text;
}

} // namespace aksharanet
