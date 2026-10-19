#pragma once

#include <string>
#include <vector>

namespace awnn {

struct BufferView {
    void* data = nullptr;
    unsigned int bytes = 0;
};

struct TensorView {
    float* data = nullptr;
    unsigned int elements = 0;
    int index = -1;
};

struct RunProfile {
    double bind_ms = 0.0;
    double run_ms = 0.0;
    double total_ms = 0.0;
};

class Session {
public:
    Session();
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool load(const std::string& model_path);
    void unload();
    bool is_loaded() const;

    int input_count() const;
    int output_count() const;

    bool input_buffer(int index, BufferView* view);
    bool copy_input(int index, const void* data, unsigned int bytes);

    bool run(RunProfile* profile = nullptr);
    bool output_tensors(std::vector<TensorView>* outputs);
    bool output_tensor(int index, TensorView* output);

private:
    class Impl;
    Impl* impl_;
};

}  // namespace awnn
