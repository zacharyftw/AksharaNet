#pragma once

#include <functional>
#include <string>
#include <vector>

namespace aksharanet {

struct Hypothesis {
    std::vector<int> tokens;
    float score = 0.0f;
};

// LM scoring callback: takes partial text, returns log-prob to add
using LMScorer = std::function<float(const std::string&)>;

class Decoder {
public:
    Decoder(std::vector<std::string> vocab, int blank_id, int eos_id,
            int beam_width = 20);

    // Set LM scorer for shallow fusion (weight controls LM influence)
    void set_lm_scorer(LMScorer scorer, float lm_weight = 0.5f);

    // Greedy decode: argmax at each position, collapse blanks
    std::string greedy_decode(const std::vector<float>& logits, int vocab_size);

    // Beam search decode (with optional LM fusion)
    std::string decode(const std::vector<float>& logits, int vocab_size);

private:
    std::vector<std::string> vocab_;
    int blank_id_;
    int eos_id_;
    int beam_width_;
    LMScorer lm_scorer_;
    float lm_weight_ = 0.0f;

    static std::vector<float> log_softmax(const float* logits, int size);
    std::string tokens_to_string(const std::vector<int>& tokens) const;
};

} // namespace aksharanet
