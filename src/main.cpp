#include <aksharanet/pipeline/pipeline.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: aksharanet <image_path>\n";
        return 1;
    }

    aksharanet::PipelineConfig config;
    aksharanet::Pipeline pipeline(config);

    std::string result = pipeline.run(argv[1]);
    std::cout << result << "\n";
    return 0;
}
