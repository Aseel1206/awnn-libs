#pragma once

#include <string>

#include <opencv2/core/core.hpp>

namespace awnn {
namespace vision {

struct LetterboxInfo {
    int target_width = 0;
    int target_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_right = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    float scale = 1.0f;
};

struct PreprocessProfile {
    double imread_ms = 0.0;
    double preprocess_ms = 0.0;
    double total_ms = 0.0;
};

bool load_bgr_image(const std::string& image_path, cv::Mat* image);

bool fill_rgb_u8_letterbox_from_mat(
    const cv::Mat& bgr_image,
    int target_width,
    int target_height,
    void* output_buffer,
    unsigned int output_bytes,
    LetterboxInfo* letterbox = nullptr,
    PreprocessProfile* profile = nullptr);

bool fill_rgb_u8_letterbox_from_path(
    const std::string& image_path,
    int target_width,
    int target_height,
    void* output_buffer,
    unsigned int output_bytes,
    LetterboxInfo* letterbox = nullptr,
    PreprocessProfile* profile = nullptr);

}  // namespace vision
}  // namespace awnn
