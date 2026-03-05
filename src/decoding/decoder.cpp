#include <aksharanet/decoding/decoder.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace aksharanet {

Decoder::Decoder(std::vector<std::string> vocab, int blank_id, int eos_id,
                 int beam_width)
    : vocab_(std::move(vocab)),
      blank_id_(blank_id),
      eos_id_(eos_id),
      beam_width_(beam_width) {}

void Decoder::set_lm_scorer(LMScorer scorer, float lm_weight) {
    lm_scorer_ = std::move(scorer);
    lm_weight_ = lm_weight;
}

std::vector<float> Decoder::log_softmax(const float* logits, int size) {
    float max_val = *std::max_element(logits, logits + size);

    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
        sum += std::exp(logits[i] - max_val);
    float log_sum = max_val + std::log(sum);

    std::vector<float> result(size);
    for (int i = 0; i < size; ++i)
        result[i] = logits[i] - log_sum;
    return result;
}

std::string Decoder::tokens_to_string(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(vocab_.size()))
            result += vocab_[id];
    }
    return result;
}

std::string Decoder::greedy_decode(const std::vector<float>& logits,
                                   int vocab_size) {
    if (logits.empty() || vocab_size <= 0)
        return {};

    int seq_len = static_cast<int>(logits.size()) / vocab_size;
    std::vector<int> tokens;
    int prev_token = -1;

    for (int t = 0; t < seq_len; ++t) {
        const float* step = logits.data() + t * vocab_size;
        int best = static_cast<int>(
            std::max_element(step, step + vocab_size) - step);

        // Skip blank tokens and repeated tokens (CTC collapse)
        if (best != blank_id_ && best != prev_token) {
            if (best == eos_id_)
                break;
            tokens.push_back(best);
        }
        prev_token = best;
    }

    return tokens_to_string(tokens);
}

std::string Decoder::decode(const std::vector<float>& logits, int vocab_size) {
    if (logits.empty() || vocab_size <= 0)
        return {};

    int seq_len = static_cast<int>(logits.size()) / vocab_size;

    // Start with one empty hypothesis
    std::vector<Hypothesis> beams(1);

    for (int t = 0; t < seq_len; ++t) {
        auto log_probs = log_softmax(logits.data() + t * vocab_size, vocab_size);

        std::vector<Hypothesis> candidates;
        candidates.reserve(beams.size() * vocab_size);

        for (const auto& beam : beams) {
            for (int v = 0; v < vocab_size; ++v) {
                Hypothesis h;
                h.score = beam.score + log_probs[v];

                if (v == blank_id_) {
                    // Blank: keep existing tokens, just add score
                    h.tokens = beam.tokens;
                } else if (v == eos_id_) {
                    // EOS: finalise this beam
                    h.tokens = beam.tokens;
                } else if (!beam.tokens.empty() && beam.tokens.back() == v) {
                    // Repeated token after non-blank: collapse (CTC)
                    h.tokens = beam.tokens;
                } else {
                    h.tokens = beam.tokens;
                    h.tokens.push_back(v);

                    // LM shallow fusion: add weighted LM score for new token
                    if (lm_scorer_ && lm_weight_ > 0.0f) {
                        std::string partial = tokens_to_string(h.tokens);
                        h.score += lm_weight_ * lm_scorer_(partial);
                    }
                }

                candidates.push_back(std::move(h));
            }
        }

        // Deduplicate: keep highest-scoring hypothesis per unique token sequence
        std::unordered_map<std::string, int> seen;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            std::string key;
            for (int id : candidates[i].tokens)
                key += std::to_string(id) + ",";

            auto it = seen.find(key);
            if (it == seen.end()) {
                seen[key] = i;
            } else if (candidates[i].score > candidates[it->second].score) {
                it->second = i;
            }
        }

        // Collect deduplicated candidates
        std::vector<Hypothesis> unique;
        unique.reserve(seen.size());
        for (auto& [_, idx] : seen)
            unique.push_back(std::move(candidates[idx]));

        // Prune to beam width
        std::partial_sort(
            unique.begin(),
            unique.begin() + std::min(beam_width_, static_cast<int>(unique.size())),
            unique.end(),
            [](const Hypothesis& a, const Hypothesis& b) {
                return a.score > b.score;
            });

        beams.assign(
            unique.begin(),
            unique.begin() + std::min(beam_width_, static_cast<int>(unique.size())));
    }

    if (beams.empty())
        return {};

    return tokens_to_string(beams[0].tokens);
}

} // namespace aksharanet
