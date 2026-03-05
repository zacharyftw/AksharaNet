#include <gtest/gtest.h>
#include <aksharanet/decoding/decoder.hpp>
#include <cmath>
#include <vector>

// Helper: build a logits matrix where token `id` wins at each step
// vocab: {"<blank>", "<eos>", "a", "b", "c"}
// blank_id=0, eos_id=1
static std::vector<float> make_logits(const std::vector<int>& winning_ids,
                                      int vocab_size, float win_score = 10.0f) {
    std::vector<float> logits(winning_ids.size() * vocab_size, 0.0f);
    for (size_t t = 0; t < winning_ids.size(); ++t)
        logits[t * vocab_size + winning_ids[t]] = win_score;
    return logits;
}

class DecodingTest : public ::testing::Test {
protected:
    // Vocabulary: 0=blank, 1=eos, 2="a", 3="b", 4="c"
    std::vector<std::string> vocab = {"", "", "a", "b", "c"};
    int blank_id = 0;
    int eos_id   = 1;
    int vocab_size = 5;
};

TEST_F(DecodingTest, EmptyLogitsReturnsEmpty) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    EXPECT_TRUE(d.decode({}, vocab_size).empty());
    EXPECT_TRUE(d.greedy_decode({}, vocab_size).empty());
}

TEST_F(DecodingTest, GreedyDecodesSimpleSequence) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    // Winning tokens: a, b, c → "abc"
    auto logits = make_logits({2, 3, 4}, vocab_size);
    EXPECT_EQ(d.greedy_decode(logits, vocab_size), "abc");
}

TEST_F(DecodingTest, GreedyCollapsesRepeats) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    // Winning tokens: a, a, b → "ab" (repeated a collapsed)
    auto logits = make_logits({2, 2, 3}, vocab_size);
    EXPECT_EQ(d.greedy_decode(logits, vocab_size), "ab");
}

TEST_F(DecodingTest, GreedySkipsBlanks) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    // Winning tokens: a, blank, b → "ab"
    auto logits = make_logits({2, 0, 3}, vocab_size);
    EXPECT_EQ(d.greedy_decode(logits, vocab_size), "ab");
}

TEST_F(DecodingTest, GreedyStopsAtEOS) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    // Winning tokens: a, eos, c → "a" (stops at eos)
    auto logits = make_logits({2, 1, 4}, vocab_size);
    EXPECT_EQ(d.greedy_decode(logits, vocab_size), "a");
}

TEST_F(DecodingTest, BeamSearchDecodesSimpleSequence) {
    aksharanet::Decoder d(vocab, blank_id, eos_id, 3);
    auto logits = make_logits({2, 3, 4}, vocab_size);
    EXPECT_EQ(d.decode(logits, vocab_size), "abc");
}

TEST_F(DecodingTest, BeamSearchCollapsesRepeats) {
    aksharanet::Decoder d(vocab, blank_id, eos_id, 3);
    auto logits = make_logits({2, 2, 3}, vocab_size);
    EXPECT_EQ(d.decode(logits, vocab_size), "ab");
}

TEST_F(DecodingTest, BeamSearchSkipsBlanks) {
    aksharanet::Decoder d(vocab, blank_id, eos_id, 3);
    auto logits = make_logits({2, 0, 3}, vocab_size);
    EXPECT_EQ(d.decode(logits, vocab_size), "ab");
}

TEST_F(DecodingTest, BeamWidthOneMatchesGreedy) {
    aksharanet::Decoder d(vocab, blank_id, eos_id, 1);
    auto logits = make_logits({2, 3, 4, 2}, vocab_size);
    std::string greedy = d.greedy_decode(logits, vocab_size);
    std::string beam   = d.decode(logits, vocab_size);
    EXPECT_EQ(greedy, beam);
}

TEST_F(DecodingTest, BlankBetweenSameTokenAllowsRepeat) {
    aksharanet::Decoder d(vocab, blank_id, eos_id);
    // a, blank, a → "aa" (blank separates two 'a' tokens)
    auto logits = make_logits({2, 0, 2}, vocab_size);
    EXPECT_EQ(d.greedy_decode(logits, vocab_size), "aa");
}
