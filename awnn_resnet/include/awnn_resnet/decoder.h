#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace awnn {
namespace resnet {

struct RawTensorView {
    const float* data = nullptr;
    unsigned int elements = 0;
    int index = -1;
};

struct ResnetModelConfig {
    int class_count = 1000;
    int top_k = 5;
    std::vector<std::string> class_names;
    bool binary_single_output = false;
    bool apply_sigmoid_to_binary_output = false;
    int binary_positive_class_id = 1;
};

struct ResnetPrediction {
    int class_id = -1;
    float score = 0.0f;
    std::string label;
};

struct ResnetDecodeResult {
    std::vector<ResnetPrediction> predictions;
    double decode_ms = 0.0;
};

class ResnetDecoder {
public:
    explicit ResnetDecoder(ResnetModelConfig config = {});

    bool decode(const std::vector<RawTensorView>& outputs, ResnetDecodeResult* result) const;

    bool render_overlay(
        const cv::Mat& input_image,
        const ResnetDecodeResult& result,
        cv::Mat* output_image) const;

    bool render_overlay_in_place(
        cv::Mat* image,
        const ResnetDecodeResult& result) const;

    const ResnetModelConfig& config() const;

private:
    ResnetModelConfig config_;
};

}  // namespace resnet
}  // namespace awnn
