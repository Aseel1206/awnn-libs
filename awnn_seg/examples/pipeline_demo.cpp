#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "awnn_inference/session.h"
#include "awnn_seg/decoder.h"
#include "awnn_vision/image_preprocessor.h"

namespace {

const char* kUsage =
    "awnn_seg_pipeline_demo -m <model.nb> -i <image.jpg> "
    "[-o output.png] [--overlay copy|inplace|none]\n";

std::vector<awnn::seg::RawTensorView> to_seg_views(const std::vector<awnn::TensorView>& outputs)
{
    std::vector<awnn::seg::RawTensorView> views;
    views.reserve(outputs.size());
    for (const auto& output : outputs) {
        awnn::seg::RawTensorView view;
        view.data = output.data;
        view.elements = output.elements;
        view.index = output.index;
        views.push_back(view);
    }
    return views;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string model_path;
    std::string image_path;
    std::string output_path = "overlay_output.png";
    std::string overlay_mode = "copy";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            image_path = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--overlay") && i + 1 < argc) {
            overlay_mode = argv[++i];
        } else if (!strcmp(argv[i], "-h")) {
            printf("%s", kUsage);
            return 0;
        }
    }

    if (model_path.empty() || image_path.empty()) {
        printf("%s", kUsage);
        return 1;
    }

    awnn::Session session;
    if (!session.load(model_path)) {
        printf("Failed to load model: %s\n", model_path.c_str());
        return 2;
    }

    awnn::BufferView input;
    if (!session.input_buffer(0, &input)) {
        printf("Failed to get model input buffer.\n");
        return 3;
    }

    awnn::vision::LetterboxInfo letterbox;
    awnn::vision::PreprocessProfile preprocess_profile;
    if (!awnn::vision::fill_rgb_u8_letterbox_from_path(
            image_path,
            640,
            640,
            input.data,
            input.bytes,
            &letterbox,
            &preprocess_profile)) {
        printf("Failed to preprocess image: %s\n", image_path.c_str());
        return 4;
    }

    awnn::RunProfile run_profile;
    if (!session.run(&run_profile)) {
        printf("Inference failed.\n");
        return 5;
    }

    std::vector<awnn::TensorView> outputs;
    if (!session.output_tensors(&outputs)) {
        printf("Failed to fetch model outputs.\n");
        return 6;
    }

    cv::Mat original = cv::imread(image_path, cv::IMREAD_COLOR);
    if (original.empty()) {
        printf("Failed to reload image for decode/render: %s\n", image_path.c_str());
        return 7;
    }

    awnn::seg::SegModelConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.class_count = 1;
    config.class_names = {"mine"};

    awnn::seg::SegDecoder decoder(config);
    awnn::seg::SegDecodeResult decode_result;
    if (!decoder.decode(original, to_seg_views(outputs), &decode_result)) {
        printf("Segmentation decode failed.\n");
        return 8;
    }

    printf("Preprocess: imread %.3f ms, transform %.3f ms, total %.3f ms\n",
           preprocess_profile.imread_ms, preprocess_profile.preprocess_ms, preprocess_profile.total_ms);
    printf("Inference: bind %.3f ms, run %.3f ms, total %.3f ms\n",
           run_profile.bind_ms, run_profile.run_ms, run_profile.total_ms);
    printf("Decode: %.3f ms\n", decode_result.decode_ms);
    printf("Detections: %zu\n", decode_result.detections.size());

    for (size_t i = 0; i < decode_result.detections.size(); ++i) {
        const auto& det = decode_result.detections[i];
        printf("[%zu] class=%d score=%.3f box=[%.1f %.1f %.1f %.1f] mask=%dx%d\n",
               i,
               det.class_id,
               det.score,
               det.box.x,
               det.box.y,
               det.box.width,
               det.box.height,
               det.mask.cols,
               det.mask.rows);
    }

    if (overlay_mode == "none") {
        return 0;
    }

    cv::Mat rendered;
    if (overlay_mode == "copy") {
        if (!decoder.render_overlay(original, decode_result, &rendered)) {
            printf("Overlay rendering failed.\n");
            return 9;
        }
        if (!cv::imwrite(output_path, rendered)) {
            printf("Failed to save overlay image: %s\n", output_path.c_str());
            return 10;
        }
        printf("Saved overlay copy to %s\n", output_path.c_str());
    } else if (overlay_mode == "inplace") {
        if (!decoder.render_overlay_in_place(&original, decode_result)) {
            printf("In-place overlay failed.\n");
            return 11;
        }
        if (!cv::imwrite(output_path, original)) {
            printf("Failed to save in-place overlay image: %s\n", output_path.c_str());
            return 12;
        }
        printf("Saved in-place overlay result to %s\n", output_path.c_str());
    } else {
        printf("Unknown overlay mode: %s\n", overlay_mode.c_str());
        return 13;
    }

    return 0;
}
