#include "awnn_resnet/decoder.h"

#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace awnn {
namespace resnet {
namespace {

double elapsed_ms(const timespec& start, const timespec& end)
{
    const double start_ms = static_cast<double>(start.tv_sec) * 1000.0 +
                            static_cast<double>(start.tv_nsec) / 1000000.0;
    const double end_ms = static_cast<double>(end.tv_sec) * 1000.0 +
                          static_cast<double>(end.tv_nsec) / 1000000.0;
    return end_ms - start_ms;
}

bool validate_outputs(const std::vector<RawTensorView>& outputs)
{
    return !outputs.empty() && outputs[0].data != nullptr && outputs[0].elements > 0;
}

std::string class_label(const ResnetModelConfig& config, int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(config.class_names.size())) {
        return config.class_names[class_id];
    }
    return std::to_string(class_id);
}

float clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

ResnetDecoder::ResnetDecoder(ResnetModelConfig config) : config_(std::move(config)) {}

bool ResnetDecoder::decode(const std::vector<RawTensorView>& outputs, ResnetDecodeResult* result) const
{
    if (result == nullptr || !validate_outputs(outputs)) {
        return false;
    }

    timespec begin {};
    timespec end {};
    clock_gettime(CLOCK_MONOTONIC, &begin);

    const RawTensorView& output = outputs[0];

    result->predictions.clear();

    if (config_.binary_single_output && output.elements == 1) {
        float positive_score = output.data[0];
        if (config_.apply_sigmoid_to_binary_output || positive_score < 0.0f || positive_score > 1.0f) {
            positive_score = sigmoid(positive_score);
        }
        positive_score = clamp01(positive_score);

        const int positive_class_id = config_.binary_positive_class_id;
        const int negative_class_id = (positive_class_id == 0) ? 1 : 0;
        const float negative_score = 1.0f - positive_score;

        ResnetPrediction positive_pred;
        positive_pred.class_id = positive_class_id;
        positive_pred.score = positive_score;
        positive_pred.label = class_label(config_, positive_class_id);

        ResnetPrediction negative_pred;
        negative_pred.class_id = negative_class_id;
        negative_pred.score = negative_score;
        negative_pred.label = class_label(config_, negative_class_id);

        result->predictions.reserve(2);
        if (positive_pred.score >= negative_pred.score) {
            result->predictions.push_back(std::move(positive_pred));
            result->predictions.push_back(std::move(negative_pred));
        } else {
            result->predictions.push_back(std::move(negative_pred));
            result->predictions.push_back(std::move(positive_pred));
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        result->decode_ms = elapsed_ms(begin, end);
        return true;
    }

    const int class_count = std::min(config_.class_count, static_cast<int>(output.elements));
    if (class_count <= 0) {
        return false;
    }

    std::vector<int> indices(static_cast<size_t>(class_count));
    std::iota(indices.begin(), indices.end(), 0);

    const int k = std::min(config_.top_k, class_count);
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&](int a, int b) { return output.data[a] > output.data[b]; });

    result->predictions.reserve(static_cast<size_t>(k));

    for (int i = 0; i < k; ++i) {
        const int class_id = indices[i];
        ResnetPrediction pred;
        pred.class_id = class_id;
        pred.score = output.data[class_id];
        pred.label = class_label(config_, class_id);
        result->predictions.push_back(std::move(pred));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    result->decode_ms = elapsed_ms(begin, end);
    return true;
}

bool ResnetDecoder::render_overlay(
    const cv::Mat& input_image,
    const ResnetDecodeResult& result,
    cv::Mat* output_image) const
{
    if (output_image == nullptr || input_image.empty()) {
        return false;
    }

    *output_image = input_image.clone();
    return render_overlay_in_place(output_image, result);
}

bool ResnetDecoder::render_overlay_in_place(
    cv::Mat* image,
    const ResnetDecodeResult& result) const
{
    if (image == nullptr || image->empty()) {
        return false;
    }

    if (result.predictions.empty()) {
        return true;
    }

    const ResnetPrediction& top = result.predictions.front();
    char text[512];
    snprintf(text, sizeof(text), "%s %.4f", top.label.c_str(), top.score);

    int baseline = 0;
    const cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    const int x = 10;
    const int y = std::max(10 + label_size.height, 20);

    cv::rectangle(
        *image,
        cv::Rect(cv::Point(x - 5, y - label_size.height - 5),
                 cv::Size(label_size.width + 10, label_size.height + baseline + 10)),
        cv::Scalar(255, 255, 255),
        -1);
    cv::putText(*image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    return true;
}

const ResnetModelConfig& ResnetDecoder::config() const
{
    return config_;
}

}  // namespace resnet
}  // namespace awnn
