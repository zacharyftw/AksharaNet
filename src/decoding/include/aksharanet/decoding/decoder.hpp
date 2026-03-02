#pragma once

#include <string>
#include <vector>

namespace aksharanet {

class Decoder {
public:
    explicit Decoder(int beam_width = 20);
    std::string decode(const std::vector<float>& logits);
};

} // namespace aksharanet
