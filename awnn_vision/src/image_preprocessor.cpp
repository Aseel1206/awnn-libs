#include "awnn_vision/image_preprocessor.h"

#include <stdio.h>
#include <time.h>

#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace awnn {
namespace vision {
namespace {

double elapsed_ms(const timespec& start, const timespec& end)
{
    const double start_ms = static_cast<double>(start.tv_sec) * 1000.0 +
                            static_cast<double>(start.tv_nsec) / 1000000.0;
    const double end_ms = static_cast<double>(end.tv_sec) * 1000.0 +
                          static_cast<double>(end.tv_nsec) / 1000000.0;
    return end_ms - start_ms;
}

}  // namespace

bool load_bgr_image(const std::string& image_path, cv::Mat* image)
{
    if (image == nullptr) {
        return false;
    }

    *image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image->empty()) {
        fprintf(stderr, "awnn_vision: cv::imread failed for %s\n", image_path.c_str());
        return false;
    }

    return true;
}

bool fill_rgb_u8_letterbox_from_mat(
    const cv::Mat& bgr_image,
    int target_width,
    int target_height,
    void* output_buffer,
    unsigned int output_bytes,
    LetterboxInfo* letterbox,
    PreprocessProfile* profile)
{
    if (bgr_image.empty() || output_buffer == nullptr || target_width <= 0 || target_height <= 0) {
        return false;
    }

    const unsigned int required_bytes = static_cast<unsigned int>(target_width * target_height * 3);
    if (output_bytes < required_bytes) {
        fprintf(stderr, "awnn_vision: output buffer too small. need=%u have=%u\n", required_bytes, output_bytes);
        return false;
    }

    timespec preprocess_begin {};
    timespec preprocess_end {};
    clock_gettime(CLOCK_MONOTONIC, &preprocess_begin);

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    float scale = 1.0f;
    if ((target_height * 1.0f / rgb_image.rows) < (target_width * 1.0f / rgb_image.cols)) {
        scale = target_height * 1.0f / rgb_image.rows;
    } else {
        scale = target_width * 1.0f / rgb_image.cols;
    }

    const int resized_width = static_cast<int>(std::round(scale * rgb_image.cols));
    const int resized_height = static_cast<int>(std::round(scale * rgb_image.rows));

    const float dh = (target_height - resized_height) / 2.0f;
    const float dw = (target_width - resized_width) / 2.0f;

    const int top = static_cast<int>(std::round(dh - 0.1f));
    const int bottom = static_cast<int>(std::round(dh + 0.1f));
    const int left = static_cast<int>(std::round(dw - 0.1f));
    const int right = static_cast<int>(std::round(dw + 0.1f));

    cv::resize(rgb_image, rgb_image, cv::Size(resized_width, resized_height));

    cv::Mat output_mat(target_height, target_width, CV_8UC3, output_buffer);
    cv::copyMakeBorder(
        rgb_image,
        output_mat,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    clock_gettime(CLOCK_MONOTONIC, &preprocess_end);

    if (letterbox != nullptr) {
        letterbox->target_width = target_width;
        letterbox->target_height = target_height;
        letterbox->resized_width = resized_width;
        letterbox->resized_height = resized_height;
        letterbox->pad_left = left;
        letterbox->pad_right = right;
        letterbox->pad_top = top;
        letterbox->pad_bottom = bottom;
        letterbox->scale = scale;
    }

    if (profile != nullptr) {
        profile->preprocess_ms = elapsed_ms(preprocess_begin, preprocess_end);
    }

    return true;
}

bool fill_rgb_u8_letterbox_from_path(
    const std::string& image_path,
    int target_width,
    int target_height,
    void* output_buffer,
    unsigned int output_bytes,
    LetterboxInfo* letterbox,
    PreprocessProfile* profile)
{
    timespec total_begin {};
    timespec imread_begin {};
    timespec imread_end {};
    timespec total_end {};
    clock_gettime(CLOCK_MONOTONIC, &total_begin);
    clock_gettime(CLOCK_MONOTONIC, &imread_begin);

    cv::Mat bgr_image;
    if (!load_bgr_image(image_path, &bgr_image)) {
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &imread_end);

    PreprocessProfile local_profile;
    if (!fill_rgb_u8_letterbox_from_mat(
            bgr_image,
            target_width,
            target_height,
            output_buffer,
            output_bytes,
            letterbox,
            &local_profile)) {
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);

    if (profile != nullptr) {
        profile->imread_ms = elapsed_ms(imread_begin, imread_end);
        profile->preprocess_ms = local_profile.preprocess_ms;
        profile->total_ms = elapsed_ms(total_begin, total_end);
    }

    return true;
}

}  // namespace vision
}  // namespace awnn
