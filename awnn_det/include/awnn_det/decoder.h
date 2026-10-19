#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace awnn {
namespace det {

struct RawTensorView {
    const float* data = nullptr;
    unsigned int elements = 0;
    int index = -1;
};

struct DetModelConfig {
    int input_width = 640;
    int input_height = 640;
    int class_count = 80;
    float score_threshold = 0.4f;
    float nms_threshold = 0.45f;
    std::vector<std::string> class_names;
};

struct DetDetection {
    cv::Rect2f box;
    int class_id = -1;
    float score = 0.0f;
};

struct DetDecodeResult {
    std::vector<DetDetection> detections;
    double decode_ms = 0.0;
};

class DetDecoder {
public:
    explicit DetDecoder(DetModelConfig config = {});

    bool decode(
        const cv::Mat& bgr_image,
        const std::vector<RawTensorView>& outputs,
        DetDecodeResult* result) const;

    bool render_overlay(
        const cv::Mat& input_image,
        const DetDecodeResult& result,
        cv::Mat* output_image) const;

    bool render_overlay_in_place(
        cv::Mat* image,
        const DetDecodeResult& result) const;

    const DetModelConfig& config() const;

private:
    DetModelConfig config_;
};

}  // namespace det
}  // namespace awnn
