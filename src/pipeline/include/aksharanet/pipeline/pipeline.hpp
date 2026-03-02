#pragma once

#include <string>

namespace aksharanet {

struct PipelineConfig {
    std::string model_path;
    std::string lm_path;
    int beam_width   = 20;
    bool enable_cuda = false;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    std::string run(const std::string& image_path);
};

} // namespace aksharanet
