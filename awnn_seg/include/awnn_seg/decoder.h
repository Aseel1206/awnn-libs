#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace awnn {
namespace seg {

struct RawTensorView {
    const float* data = nullptr;
    unsigned int elements = 0;
    int index = -1;
};

struct SegModelConfig {
    int input_width = 640;
    int input_height = 640;
    int class_count = 1;
    float score_threshold = 0.4f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int mask_proto_channels = 32;
    int mask_stride = 4;
    std::vector<std::string> class_names {"class0"};
};

struct SegDetection {
    cv::Rect2f box;
    int class_id = -1;
    float score = 0.0f;
    cv::Mat mask;
};

struct SegDecodeResult {
    std::vector<SegDetection> detections;
    double decode_ms = 0.0;
};

class SegDecoder {
public:
    explicit SegDecoder(SegModelConfig config = {});

    bool decode(
        const cv::Mat& bgr_image,
        const std::vector<RawTensorView>& outputs,
        SegDecodeResult* result) const;

    bool render_overlay(
        const cv::Mat& input_image,
        const SegDecodeResult& result,
        cv::Mat* output_image) const;

    bool render_overlay_in_place(
        cv::Mat* image,
        const SegDecodeResult& result) const;

    const SegModelConfig& config() const;

private:
    SegModelConfig config_;
};

}  // namespace seg
}  // namespace awnn
