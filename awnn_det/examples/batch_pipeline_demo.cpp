#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "awnn_det/decoder.h"
#include "awnn_inference/session.h"
#include "awnn_vision/image_preprocessor.h"

namespace {

const char* kUsage =
    "awnn_det_batch_pipeline_demo -m <model.nb> -i <input_dir> -o <output_dir> "
    "[--overlay copy|inplace|none]\n";

std::vector<std::string> mine_names()
{
    return {
        "mine", "not mine"
    };
}

std::string to_lower(std::string value)
{
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = static_cast<char>(tolower(static_cast<unsigned char>(value[i])));
    }
    return value;
}

bool has_image_extension(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }

    const std::string ext = to_lower(path.substr(dot));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::string join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

std::string basename_only(const std::string& path)
{
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool ensure_directory(const std::string& path)
{
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path.c_str(), 0755) == 0;
}

std::vector<std::string> list_images(const std::string& input_dir)
{
    std::vector<std::string> paths;

    DIR* dir = opendir(input_dir.c_str());
    if (dir == nullptr) {
        return paths;
    }

    for (;;) {
        dirent* entry = readdir(dir);
        if (entry == nullptr) {
            break;
        }

        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        const std::string full_path = join_path(input_dir, name);
        struct stat st {};
        if (stat(full_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }

        if (has_image_extension(name)) {
            paths.push_back(full_path);
        }
    }

    closedir(dir);
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<awnn::det::RawTensorView> to_det_views(const std::vector<awnn::TensorView>& outputs)
{
    std::vector<awnn::det::RawTensorView> views;
    if (outputs.size() >= 6) {
        const int remap[6] = {0, 2, 4, 1, 3, 5};
        views.reserve(6);
        for (int i = 0; i < 6; ++i) {
            const auto& output = outputs[remap[i]];
            awnn::det::RawTensorView view;
            view.data = output.data;
            view.elements = output.elements;
            view.index = output.index;
            views.push_back(view);
        }
    } else {
        views.reserve(outputs.size());
        for (const auto& output : outputs) {
            awnn::det::RawTensorView view;
            view.data = output.data;
            view.elements = output.elements;
            view.index = output.index;
            views.push_back(view);
        }
    }
    return views;
}

bool process_one(
    awnn::Session* session,
    const awnn::det::DetDecoder& decoder,
    const std::string& image_path,
    const std::string& output_dir,
    const std::string& overlay_mode)
{
    awnn::BufferView input;
    if (!session->input_buffer(0, &input)) {
        printf("Failed to get model input buffer for %s\n", image_path.c_str());
        return false;
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
        return false;
    }

    awnn::RunProfile run_profile;
    if (!session->run(&run_profile)) {
        printf("Inference failed: %s\n", image_path.c_str());
        return false;
    }

    std::vector<awnn::TensorView> outputs;
    if (!session->output_tensors(&outputs)) {
        printf("Failed to fetch model outputs: %s\n", image_path.c_str());
        return false;
    }

    cv::Mat original = cv::imread(image_path, cv::IMREAD_COLOR);
    if (original.empty()) {
        printf("Failed to reload image for decode/render: %s\n", image_path.c_str());
        return false;
    }

    std::sort(outputs.begin(), outputs.end(), [](const auto& a, const auto& b) {
        return a.index < b.index;
    });

    awnn::det::DetDecodeResult decode_result;
    if (!decoder.decode(original, to_det_views(outputs), &decode_result)) {
        printf("Detection decode failed: %s\n", image_path.c_str());
        return false;
    }

    printf("%s\n", image_path.c_str());
    printf("  preprocess total %.3f ms, inference total %.3f ms, decode %.3f ms, detections %zu\n",
           preprocess_profile.total_ms, run_profile.total_ms, decode_result.decode_ms, decode_result.detections.size());

    for (size_t i = 0; i < decode_result.detections.size(); ++i) {
        const auto& det = decode_result.detections[i];
        const std::string label =
            (det.class_id >= 0 && det.class_id < static_cast<int>(decoder.config().class_names.size()))
                ? decoder.config().class_names[det.class_id]
                : std::to_string(det.class_id);
        printf("  [%zu] class=%d(%s) score=%.3f box=[%.1f %.1f %.1f %.1f]\n",
               i, det.class_id, label.c_str(), det.score, det.box.x, det.box.y, det.box.width, det.box.height);
    }

    if (overlay_mode == "none") {
        return true;
    }

    const std::string output_path = join_path(output_dir, basename_only(image_path));
    cv::Mat rendered;
    if (overlay_mode == "copy") {
        if (!decoder.render_overlay(original, decode_result, &rendered)) {
            printf("Overlay rendering failed: %s\n", image_path.c_str());
            return false;
        }
        if (!cv::imwrite(output_path, rendered)) {
            printf("Failed to save overlay image: %s\n", output_path.c_str());
            return false;
        }
    } else if (overlay_mode == "inplace") {
        if (!decoder.render_overlay_in_place(&original, decode_result)) {
            printf("In-place overlay failed: %s\n", image_path.c_str());
            return false;
        }
        if (!cv::imwrite(output_path, original)) {
            printf("Failed to save in-place overlay image: %s\n", output_path.c_str());
            return false;
        }
    } else {
        printf("Unknown overlay mode: %s\n", overlay_mode.c_str());
        return false;
    }

    printf("  saved %s\n", output_path.c_str());
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string model_path;
    std::string input_dir;
    std::string output_dir;
    std::string overlay_mode = "copy";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (!strcmp(argv[i], "--overlay") && i + 1 < argc) {
            overlay_mode = argv[++i];
        } else if (!strcmp(argv[i], "-h")) {
            printf("%s", kUsage);
            return 0;
        }
    }

    if (model_path.empty() || input_dir.empty() || output_dir.empty()) {
        printf("%s", kUsage);
        return 1;
    }

    if (!ensure_directory(output_dir)) {
        printf("Failed to create or access output directory: %s\n", output_dir.c_str());
        return 2;
    }

    const std::vector<std::string> image_paths = list_images(input_dir);
    if (image_paths.empty()) {
        printf("No images found in: %s\n", input_dir.c_str());
        return 3;
    }

    awnn::Session session;
    if (!session.load(model_path)) {
        printf("Failed to load model: %s\n", model_path.c_str());
        return 4;
    }

    awnn::det::DetModelConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.class_count = 2;
    config.score_threshold = 0.4f;
    config.nms_threshold = 0.45f;
    config.class_names = mine_names();

    awnn::det::DetDecoder decoder(config);

    size_t success_count = 0;
    for (size_t i = 0; i < image_paths.size(); ++i) {
        if (process_one(&session, decoder, image_paths[i], output_dir, overlay_mode)) {
            ++success_count;
        }
    }

    printf("Processed %zu/%zu images successfully.\n", success_count, image_paths.size());
    return success_count == image_paths.size() ? 0 : 5;
}
