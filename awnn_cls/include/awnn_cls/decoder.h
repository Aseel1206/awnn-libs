#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace awnn {
namespace cls {

struct RawTensorView {
    const float* data = nullptr;
    unsigned int elements = 0;
    int index = -1;
};

struct ClsModelConfig {
    int class_count = 1000;
    int top_k = 5;
    std::vector<std::string> class_names;
};

struct ClsPrediction {
    int class_id = -1;
    float score = 0.0f;
    std::string label;
};

struct ClsDecodeResult {
    std::vector<ClsPrediction> predictions;
    double decode_ms = 0.0;
};

class ClsDecoder {
public:
    explicit ClsDecoder(ClsModelConfig config = {});

    bool decode(const std::vector<RawTensorView>& outputs, ClsDecodeResult* result) const;

    bool render_overlay(
        const cv::Mat& input_image,
        const ClsDecodeResult& result,
        cv::Mat* output_image) const;

    bool render_overlay_in_place(
        cv::Mat* image,
        const ClsDecodeResult& result) const;

    const ClsModelConfig& config() const;

private:
    ClsModelConfig config_;
};

}  // namespace cls
}  // namespace awnn
