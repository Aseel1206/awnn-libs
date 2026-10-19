#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "awnn_vision/image_preprocessor.h"

namespace {

const char* kUsage =
    "awnn_vision_smoketest -i <image.jpg> -w <target_width> -h <target_height>\n";

}  // namespace

int main(int argc, char** argv)
{
    std::string image_path;
    int width = 0;
    int height = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            image_path = argv[++i];
        } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            height = atoi(argv[++i]);
        }
    }

    if (image_path.empty() || width <= 0 || height <= 0) {
        printf("%s", kUsage);
        return 1;
    }

    std::vector<unsigned char> tensor(static_cast<size_t>(width * height * 3), 0);
    awnn::vision::LetterboxInfo letterbox;
    awnn::vision::PreprocessProfile profile;

    if (!awnn::vision::fill_rgb_u8_letterbox_from_path(
            image_path,
            width,
            height,
            tensor.data(),
            static_cast<unsigned int>(tensor.size()),
            &letterbox,
            &profile)) {
        printf("Preprocess failed for %s\n", image_path.c_str());
        return 2;
    }

    printf("Preprocess success.\n");
    printf("imread: %.3f ms, preprocess: %.3f ms, total: %.3f ms\n",
           profile.imread_ms, profile.preprocess_ms, profile.total_ms);
    printf("resized=%dx%d pad(lrtb)=%d,%d,%d,%d scale=%.6f\n",
           letterbox.resized_width,
           letterbox.resized_height,
           letterbox.pad_left,
           letterbox.pad_right,
           letterbox.pad_top,
           letterbox.pad_bottom,
           letterbox.scale);
    printf("tensor bytes=%zu first bytes=%u,%u,%u,%u,%u,%u\n",
           tensor.size(),
           static_cast<unsigned int>(tensor[0]),
           static_cast<unsigned int>(tensor[1]),
           static_cast<unsigned int>(tensor[2]),
           static_cast<unsigned int>(tensor[3]),
           static_cast<unsigned int>(tensor[4]),
           static_cast<unsigned int>(tensor[5]));

    return 0;
}
