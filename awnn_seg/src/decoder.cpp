#include "awnn_seg/decoder.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace awnn {
namespace seg {
namespace {

struct InternalObject {
    cv::Rect2f box;
    int class_id = -1;
    float score = 0.0f;
    std::vector<float> mask_feat;
    cv::Mat mask;
};

double elapsed_ms(const timespec& start, const timespec& end)
{
    const double start_ms = static_cast<double>(start.tv_sec) * 1000.0 +
                            static_cast<double>(start.tv_nsec) / 1000000.0;
    const double end_ms = static_cast<double>(end.tv_sec) * 1000.0 +
                          static_cast<double>(end.tv_nsec) / 1000000.0;
    return end_ms - start_ms;
}

float intersection_area(const InternalObject& a, const InternalObject& b)
{
    return (a.box & b.box).area();
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float softmax_distance(const float* src, float* dst, int length)
{
    float alpha = -FLT_MAX;
    for (int i = 0; i < length; ++i) {
        alpha = std::max(alpha, src[i]);
    }

    float denominator = 0.0f;
    float distance = 0.0f;
    for (int i = 0; i < length; ++i) {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
        distance += i * dst[i];
    }

    return distance;
}

void sort_by_score(std::vector<InternalObject>* objects)
{
    std::sort(objects->begin(), objects->end(), [](const InternalObject& a, const InternalObject& b) {
        return a.score > b.score;
    });
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
            const float inter_area = intersection_area(a, b);
            const float union_area = areas[i] + areas[picked_index] - inter_area;
            if (union_area > 0.0f && inter_area / union_area > nms_threshold) {
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
    const float* feat_mask,
    const SegModelConfig& config,
    std::vector<InternalObject>* objects)
{
    const int num_grid_x = config.input_width / stride;
    const int num_grid_y = config.input_height / stride;
    const int num_grid_size = num_grid_x * num_grid_y;
    const int reg_max_1 = 16;
    float distances[16];

    cv::Mat out_grid(reg_max_1 * 4, num_grid_size, CV_32FC1, const_cast<float*>(feat_grid));
    cv::Mat out_score(config.class_count, num_grid_size, CV_32FC1, const_cast<float*>(feat_score));
    cv::Mat out_mask(config.mask_proto_channels, num_grid_size, CV_32FC1, const_cast<float*>(feat_mask));

    cv::transpose(out_grid, out_grid);
    cv::transpose(out_score, out_score);
    cv::transpose(out_mask, out_mask);

    for (int y = 0; y < num_grid_y; ++y) {
        for (int x = 0; x < num_grid_x; ++x) {
            const int grid_index = y * num_grid_x + x;
            const float* pred_score = reinterpret_cast<float*>(out_score.data) + grid_index * config.class_count;

            int class_id = 0;
            float score = sigmoid(pred_score[0]);
            for (int c = 1; c < config.class_count; ++c) {
                const float candidate = sigmoid(pred_score[c]);
                if (candidate > score) {
                    score = candidate;
                    class_id = c;
                }
            }

            if (score < config.score_threshold) {
                continue;
            }

            const float* pred_grid = reinterpret_cast<float*>(out_grid.data) + grid_index * reg_max_1 * 4;
            const float* pred_mask = reinterpret_cast<float*>(out_mask.data) + grid_index * config.mask_proto_channels;

            float x0 = x + 0.5f - softmax_distance(pred_grid, distances, 16);
            float y0 = y + 0.5f - softmax_distance(pred_grid + 16, distances, 16);
            float x1 = x + 0.5f + softmax_distance(pred_grid + 32, distances, 16);
            float y1 = y + 0.5f + softmax_distance(pred_grid + 48, distances, 16);

            x0 *= stride;
            y0 *= stride;
            x1 *= stride;
            y1 *= stride;

            const float width = x1 - x0;
            const float height = y1 - y0;
            if (!(width > 0.0f) || !(height > 0.0f)) {
                continue;
            }

            InternalObject object;
            object.box = cv::Rect2f(x0, y0, width, height);
            object.class_id = class_id;
            object.score = score;
            object.mask_feat.resize(config.mask_proto_channels);
            memcpy(object.mask_feat.data(), pred_mask, sizeof(float) * config.mask_proto_channels);
            objects->push_back(std::move(object));
        }
    }
}

bool validate_outputs(const std::vector<RawTensorView>& outputs)
{
    return outputs.size() >= 10 &&
           outputs[0].data != nullptr && outputs[1].data != nullptr &&
           outputs[2].data != nullptr && outputs[3].data != nullptr &&
           outputs[4].data != nullptr && outputs[5].data != nullptr &&
           outputs[6].data != nullptr && outputs[7].data != nullptr &&
           outputs[8].data != nullptr && outputs[9].data != nullptr;
}

std::vector<cv::Scalar> palette()
{
    return {
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(96, 125, 139)
    };
}

}  // namespace

SegDecoder::SegDecoder(SegModelConfig config) : config_(std::move(config))
{
    if (config_.class_names.empty()) {
        config_.class_names.resize(std::max(config_.class_count, 1), "class0");
    }
}

bool SegDecoder::decode(
    const cv::Mat& bgr_image,
    const std::vector<RawTensorView>& outputs,
    SegDecodeResult* result) const
{
    if (result == nullptr || bgr_image.empty() || !validate_outputs(outputs)) {
        return false;
    }

    timespec begin {};
    timespec end {};
    clock_gettime(CLOCK_MONOTONIC, &begin);

    std::vector<InternalObject> proposals;
    proposals.reserve(512);

    generate_proposals(8, outputs[0].data, outputs[1].data, outputs[2].data, config_, &proposals);
    generate_proposals(16, outputs[3].data, outputs[4].data, outputs[5].data, config_, &proposals);
    generate_proposals(32, outputs[6].data, outputs[7].data, outputs[8].data, config_, &proposals);

    sort_by_score(&proposals);

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

    const int proto_h = config_.input_height / config_.mask_stride;
    const int proto_w = config_.input_width / config_.mask_stride;
    cv::Mat proto_mat(config_.mask_proto_channels, proto_h * proto_w, CV_32FC1, const_cast<float*>(outputs[9].data));

    result->detections.clear();
    result->detections.reserve(picked.size());

    for (int picked_index : picked) {
        if (picked_index < 0 || picked_index >= static_cast<int>(proposals.size())) {
            continue;
        }

        InternalObject object = proposals[picked_index];
        if (!(object.box.width > 0.0f) || !(object.box.height > 0.0f)) {
            continue;
        }

        int hstart = static_cast<int>(std::floor(object.box.y / config_.mask_stride));
        int hend = static_cast<int>(std::ceil(object.box.y / config_.mask_stride + object.box.height / config_.mask_stride));
        int wstart = static_cast<int>(std::floor(object.box.x / config_.mask_stride));
        int wend = static_cast<int>(std::ceil(object.box.x / config_.mask_stride + object.box.width / config_.mask_stride));

        hstart = std::min(std::max(hstart, 0), proto_h);
        hend = std::min(std::max(hend, 0), proto_h);
        wstart = std::min(std::max(wstart, 0), proto_w);
        wend = std::min(std::max(wend, 0), proto_w);

        const int mask_w = wend - wstart;
        const int mask_h = hend - hstart;

        cv::Mat mask;
        if (mask_w > 0 && mask_h > 0) {
            mask = cv::Mat(mask_h, mask_w, CV_32FC1, cv::Scalar(0));
            std::vector<cv::Range> roi_ranges;
            roi_ranges.push_back(cv::Range(0, 1));
            roi_ranges.push_back(cv::Range::all());
            roi_ranges.push_back(cv::Range(hstart, hend));
            roi_ranges.push_back(cv::Range(wstart, wend));

            const int proto_sizes[] = {1, config_.mask_proto_channels, proto_h, proto_w};
            cv::Mat proto_reshape = proto_mat.reshape(1, 4, proto_sizes);
            cv::Mat protos = proto_reshape(roi_ranges).clone().reshape(0, {config_.mask_proto_channels, mask_w * mask_h});
            cv::Mat mask_proposals(1, config_.mask_proto_channels, CV_32FC1, object.mask_feat.data());
            cv::Mat masks_feature = mask_proposals * protos;

            cv::exp(-masks_feature.reshape(1, {mask_h, mask_w}), mask);
            mask = 1.0f / (1.0f + mask);
        }

        float x0 = (object.box.x - (wpad / 2.0f)) * ratio_x;
        float y0 = (object.box.y - (hpad / 2.0f)) * ratio_y;
        float x1 = (object.box.x + object.box.width - (wpad / 2.0f)) * ratio_x;
        float y1 = (object.box.y + object.box.height - (hpad / 2.0f)) * ratio_y;

        x0 = std::max(std::min(x0, static_cast<float>(bgr_image.cols - 1)), 0.0f);
        y0 = std::max(std::min(y0, static_cast<float>(bgr_image.rows - 1)), 0.0f);
        x1 = std::max(std::min(x1, static_cast<float>(bgr_image.cols - 1)), 0.0f);
        y1 = std::max(std::min(y1, static_cast<float>(bgr_image.rows - 1)), 0.0f);

        const float clipped_width = x1 - x0;
        const float clipped_height = y1 - y0;
        if (!(clipped_width > 0.0f) || !(clipped_height > 0.0f)) {
            continue;
        }

        SegDetection detection;
        detection.box = cv::Rect2f(x0, y0, clipped_width, clipped_height);
        detection.class_id = object.class_id;
        detection.score = object.score;

        if (!mask.empty() && detection.box.width > 0 && detection.box.height > 0) {
            const int resized_mask_w = std::max(1, static_cast<int>(std::round(detection.box.width)));
            const int resized_mask_h = std::max(1, static_cast<int>(std::round(detection.box.height)));
            cv::resize(mask, mask, cv::Size(resized_mask_w, resized_mask_h));
            detection.mask = mask > config_.mask_threshold;
        }

        result->detections.push_back(std::move(detection));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    result->decode_ms = elapsed_ms(begin, end);
    return true;
}

bool SegDecoder::render_overlay(
    const cv::Mat& input_image,
    const SegDecodeResult& result,
    cv::Mat* output_image) const
{
    if (output_image == nullptr || input_image.empty()) {
        return false;
    }

    *output_image = input_image.clone();
    return render_overlay_in_place(output_image, result);
}

bool SegDecoder::render_overlay_in_place(
    cv::Mat* image,
    const SegDecodeResult& result) const
{
    if (image == nullptr || image->empty()) {
        return false;
    }

    cv::Mat mask_layer = image->clone();
    const std::vector<cv::Scalar> colors = palette();

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const SegDetection& detection = result.detections[i];
        const cv::Scalar& color = colors[i % colors.size()];

        if (!detection.mask.empty()) {
            const cv::Rect roi(
                static_cast<int>(detection.box.x),
                static_cast<int>(detection.box.y),
                static_cast<int>(detection.box.width),
                static_cast<int>(detection.box.height));

            if (roi.width > 0 && roi.height > 0 &&
                roi.x >= 0 && roi.y >= 0 &&
                roi.x + roi.width <= mask_layer.cols &&
                roi.y + roi.height <= mask_layer.rows) {
                mask_layer(roi).setTo(color, detection.mask);
            }
        }

        cv::rectangle(*image, detection.box, color, 2);

        std::string label = std::to_string(detection.class_id);
        if (detection.class_id >= 0 && detection.class_id < static_cast<int>(config_.class_names.size())) {
            label = config_.class_names[detection.class_id];
        }
        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", label.c_str(), detection.score * 100.0f);

        int baseline = 0;
        const cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = static_cast<int>(detection.box.x);
        int y = static_cast<int>(detection.box.y) - label_size.height - baseline;
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

    *image = 0.5 * mask_layer + 0.5 * (*image);
    return true;
}

const SegModelConfig& SegDecoder::config() const
{
    return config_;
}

}  // namespace seg
}  // namespace awnn
