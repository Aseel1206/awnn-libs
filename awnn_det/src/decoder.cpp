#include "awnn_det/decoder.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace awnn {
namespace det {
namespace {

struct InternalObject {
    cv::Rect2f box;
    int class_id = -1;
    float score = 0.0f;
};

double elapsed_ms(const timespec& start, const timespec& end)
{
    const double start_ms = static_cast<double>(start.tv_sec) * 1000.0 +
                            static_cast<double>(start.tv_nsec) / 1000000.0;
    const double end_ms = static_cast<double>(end.tv_sec) * 1000.0 +
                          static_cast<double>(end.tv_nsec) / 1000000.0;
    return end_ms - start_ms;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float intersection_area(const InternalObject& a, const InternalObject& b)
{
    return (a.box & b.box).area();
}

void nms_sorted_bboxes(
    const std::vector<InternalObject>& objects,
    float nms_threshold,
    std::vector<int>* picked)
{
    picked->clear();
    std::vector<float> areas(objects.size(), 0.0f);
    for (size_t i = 0; i < objects.size(); ++i) {
        areas[i] = objects[i].box.area();
    }

    for (size_t i = 0; i < objects.size(); ++i) {
        const InternalObject& a = objects[i];
        bool keep = true;

        for (int picked_index : *picked) {
            const InternalObject& b = objects[picked_index];
            if (a.class_id != b.class_id) {
                continue;
            }

            const float inter = intersection_area(a, b);
            const float uni = areas[i] + areas[picked_index] - inter;
            if (uni > 0.0f && inter / uni > nms_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            picked->push_back(static_cast<int>(i));
        }
    }
}

void generate_proposals(
    int stride,
    const float* feat_grid,
    const float* feat_score,
    const DetModelConfig& config,
    std::vector<InternalObject>* objects)
{
    const int num_grid_x = config.input_width / stride;
    const int num_grid_y = config.input_height / stride;
    const int num_grid_size = num_grid_x * num_grid_y;

    cv::Mat out_grid(4, num_grid_size, CV_32FC1, const_cast<float*>(feat_grid));
    cv::Mat out_score(config.class_count, num_grid_size, CV_32FC1, const_cast<float*>(feat_score));

    cv::transpose(out_grid, out_grid);
    cv::transpose(out_score, out_score);

    for (int y = 0; y < num_grid_y; ++y) {
        for (int x = 0; x < num_grid_x; ++x) {
            const int grid_index = y * num_grid_x + x;
            const float* pred_score = reinterpret_cast<float*>(out_score.data) + grid_index * config.class_count;

            int class_id = -1;
            float score = -FLT_MAX;
            for (int c = 0; c < config.class_count; ++c) {
                if (pred_score[c] > score) {
                    score = pred_score[c];
                    class_id = c;
                }
            }

            score = sigmoid(score);
            if (class_id < 0 || score < config.score_threshold) {
                continue;
            }

            const float* pred_grid = reinterpret_cast<float*>(out_grid.data) + grid_index * 4;
            float x0 = (x + 0.5f - pred_grid[0]) * stride;
            float y0 = (y + 0.5f - pred_grid[1]) * stride;
            float x1 = (x + 0.5f + pred_grid[2]) * stride;
            float y1 = (y + 0.5f + pred_grid[3]) * stride;

            const float width = x1 - x0;
            const float height = y1 - y0;
            if (!(width > 0.0f) || !(height > 0.0f)) {
                continue;
            }

            InternalObject obj;
            obj.box = cv::Rect2f(x0, y0, width, height);
            obj.class_id = class_id;
            obj.score = score;
            objects->push_back(std::move(obj));
        }
    }
}

bool validate_outputs(const std::vector<RawTensorView>& outputs)
{
    return outputs.size() >= 6 &&
           outputs[0].data != nullptr && outputs[1].data != nullptr &&
           outputs[2].data != nullptr && outputs[3].data != nullptr &&
           outputs[4].data != nullptr && outputs[5].data != nullptr;
}

std::vector<cv::Scalar> palette()
{
    return {
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255)
    };
}

}  // namespace

DetDecoder::DetDecoder(DetModelConfig config) : config_(std::move(config)) {}

bool DetDecoder::decode(
    const cv::Mat& bgr_image,
    const std::vector<RawTensorView>& outputs,
    DetDecodeResult* result) const
{
    if (result == nullptr || bgr_image.empty() || !validate_outputs(outputs)) {
        return false;
    }

    timespec begin {};
    timespec end {};
    clock_gettime(CLOCK_MONOTONIC, &begin);

    std::vector<InternalObject> proposals;
    proposals.reserve(1024);

    generate_proposals(8, outputs[0].data, outputs[3].data, config_, &proposals);
    generate_proposals(16, outputs[1].data, outputs[4].data, config_, &proposals);
    generate_proposals(32, outputs[2].data, outputs[5].data, config_, &proposals);

    std::sort(proposals.begin(), proposals.end(), [](const InternalObject& a, const InternalObject& b) {
        return a.score > b.score;
    });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, config_.nms_threshold, &picked);

    const float scale = std::min(
        config_.input_height * 1.0f / bgr_image.rows,
        config_.input_width * 1.0f / bgr_image.cols);
    const int resized_width = static_cast<int>(std::round(scale * bgr_image.cols));
    const int resized_height = static_cast<int>(std::round(scale * bgr_image.rows));
    if (resized_width <= 0 || resized_height <= 0) {
        return false;
    }

    const int wpad = config_.input_width - resized_width;
    const int hpad = config_.input_height - resized_height;
    const float ratio_x = static_cast<float>(bgr_image.cols) / resized_width;
    const float ratio_y = static_cast<float>(bgr_image.rows) / resized_height;

    result->detections.clear();
    result->detections.reserve(picked.size());

    for (int picked_index : picked) {
        if (picked_index < 0 || picked_index >= static_cast<int>(proposals.size())) {
            continue;
        }

        const InternalObject& object = proposals[picked_index];
        float x0 = (object.box.x - (wpad / 2.0f)) * ratio_x;
        float y0 = (object.box.y - (hpad / 2.0f)) * ratio_y;
        float x1 = (object.box.x + object.box.width - (wpad / 2.0f)) * ratio_x;
        float y1 = (object.box.y + object.box.height - (hpad / 2.0f)) * ratio_y;

        x0 = std::max(std::min(x0, static_cast<float>(bgr_image.cols - 1)), 0.0f);
        y0 = std::max(std::min(y0, static_cast<float>(bgr_image.rows - 1)), 0.0f);
        x1 = std::max(std::min(x1, static_cast<float>(bgr_image.cols - 1)), 0.0f);
        y1 = std::max(std::min(y1, static_cast<float>(bgr_image.rows - 1)), 0.0f);

        const float width = x1 - x0;
        const float height = y1 - y0;
        if (!(width > 0.0f) || !(height > 0.0f)) {
            continue;
        }

        DetDetection det;
        det.box = cv::Rect2f(x0, y0, width, height);
        det.class_id = object.class_id;
        det.score = object.score;
        result->detections.push_back(std::move(det));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    result->decode_ms = elapsed_ms(begin, end);
    return true;
}

bool DetDecoder::render_overlay(
    const cv::Mat& input_image,
    const DetDecodeResult& result,
    cv::Mat* output_image) const
{
    if (output_image == nullptr || input_image.empty()) {
        return false;
    }

    *output_image = input_image.clone();
    return render_overlay_in_place(output_image, result);
}

bool DetDecoder::render_overlay_in_place(
    cv::Mat* image,
    const DetDecodeResult& result) const
{
    if (image == nullptr || image->empty()) {
        return false;
    }

    const std::vector<cv::Scalar> colors = palette();
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        const auto& color = colors[i % colors.size()];

        cv::rectangle(*image, det.box, color, 2);

        std::string label = std::to_string(det.class_id);
        if (det.class_id >= 0 && det.class_id < static_cast<int>(config_.class_names.size())) {
            label = config_.class_names[det.class_id];
        }

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", label.c_str(), det.score * 100.0f);

        int baseline = 0;
        const cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = static_cast<int>(det.box.x);
        int y = static_cast<int>(det.box.y) - label_size.height - baseline;
        if (y < 0) {
            y = 0;
        }
        if (x + label_size.width > image->cols) {
            x = image->cols - label_size.width;
        }

        cv::rectangle(*image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseline)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(*image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return true;
}

const DetModelConfig& DetDecoder::config() const
{
    return config_;
}

}  // namespace det
}  // namespace awnn
