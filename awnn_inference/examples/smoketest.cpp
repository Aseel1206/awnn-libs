#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "awnn_inference/session.h"

namespace {

const char* kUsage =
    "awnn_inference_smoketest -m <model.nb> [-i input.bin] [-z]\n"
    "  -m <model.nb>   Path to NPU model\n"
    "  -i <input.bin>  Optional raw input tensor file\n"
    "  -z              Zero-fill input buffer instead of reading a file\n";

bool read_file(const std::string& path, std::vector<unsigned char>* data)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size < 0) {
        return false;
    }

    data->resize(static_cast<size_t>(size));
    if (size == 0) {
        return true;
    }

    return file.read(reinterpret_cast<char*>(data->data()), size).good();
}

}  // namespace

int main(int argc, char** argv)
{
    std::string model_path;
    std::string input_path;
    bool zero_fill = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            input_path = argv[++i];
        } else if (!strcmp(argv[i], "-z")) {
            zero_fill = true;
        } else if (!strcmp(argv[i], "-h")) {
            printf("%s", kUsage);
            return 0;
        }
    }

    if (model_path.empty() || (!zero_fill && input_path.empty())) {
        printf("%s", kUsage);
        return 1;
    }

    awnn::Session session;
    if (!session.load(model_path)) {
        printf("Failed to load model: %s\n", model_path.c_str());
        return 2;
    }

    printf("Model loaded.\n");
    printf("Inputs: %d, Outputs: %d\n", session.input_count(), session.output_count());

    awnn::BufferView input;
    if (!session.input_buffer(0, &input)) {
        printf("Failed to get input buffer 0.\n");
        return 3;
    }

    printf("Input[0] buffer: %u bytes\n", input.bytes);

    if (zero_fill) {
        memset(input.data, 0, input.bytes);
        printf("Input buffer zero-filled.\n");
    } else {
        std::vector<unsigned char> input_data;
        if (!read_file(input_path, &input_data)) {
            printf("Failed to read input file: %s\n", input_path.c_str());
            return 4;
        }

        const unsigned int copy_bytes = std::min<unsigned int>(
            input.bytes, static_cast<unsigned int>(input_data.size()));
        memcpy(input.data, input_data.data(), copy_bytes);
        printf("Copied %u/%zu bytes from %s\n", copy_bytes, input_data.size(), input_path.c_str());
    }

    awnn::RunProfile profile;
    if (!session.run(&profile)) {
        printf("Inference failed.\n");
        return 5;
    }

    printf("Bind: %.3f ms, Run: %.3f ms, Total: %.3f ms\n",
           profile.bind_ms, profile.run_ms, profile.total_ms);

    std::vector<awnn::TensorView> outputs;
    if (!session.output_tensors(&outputs)) {
        printf("Failed to read outputs.\n");
        return 6;
    }

    for (const auto& output : outputs) {
        printf("Output[%d]: elements=%u", output.index, output.elements);
        if (output.data != nullptr && output.elements > 0) {
            const unsigned int preview = std::min<unsigned int>(output.elements, 8);
            printf(", first_values=");
            for (unsigned int i = 0; i < preview; ++i) {
                printf(i == 0 ? "%.6f" : ",%.6f", output.data[i]);
            }
        }
        printf("\n");
    }

    session.unload();
    return 0;
}
