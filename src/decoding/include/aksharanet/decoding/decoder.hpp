#pragma once

#include <string>
#include <vector>

namespace aksharanet {

struct Hypothesis {
    std::vector<int> tokens;
    float score = 0.0f;
};

class Decoder {
public:
    // vocab: grapheme cluster strings indexed by token ID
    // blank_id: CTC blank token ID (collapsed during decoding)
    // eos_id: end-of-sequence token ID
    Decoder(std::vector<std::string> vocab, int blank_id, int eos_id,
            int beam_width = 20);

    // Greedy decode: argmax at each position, collapse blanks
    std::string greedy_decode(const std::vector<float>& logits, int vocab_size);

    // Beam search decode
    std::string decode(const std::vector<float>& logits, int vocab_size);

private:
    std::vector<std::string> vocab_;
    int blank_id_;
    int eos_id_;
    int beam_width_;

    // Convert log-probabilities from logits at a single time step
    static std::vector<float> log_softmax(const float* logits, int size);

    // Convert token ID sequence to string using vocabulary
    std::string tokens_to_string(const std::vector<int>& tokens) const;
};

} // namespace aksharanet
