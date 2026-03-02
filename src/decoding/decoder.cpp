#include <aksharanet/decoding/decoder.hpp>

namespace aksharanet {

Decoder::Decoder(int /*beam_width*/) {}

std::string Decoder::decode(const std::vector<float>& /*logits*/) {
    return {};
}

} // namespace aksharanet
