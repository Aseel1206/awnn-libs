#include "awnn_inference/session.h"

#include <stdio.h>
#include <time.h>

#include <memory>
#include <vector>

#include "npulib.h"

namespace awnn {
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

class Session::Impl {
public:
    bool load(const std::string& model_path)
    {
        unload();

        npu_ = std::make_unique<NpuUint>();
        network_ = std::make_unique<NetworkItem>();

        if (npu_->npu_init() != 0) {
            printf("awnn_inference: failed to initialize NPU runtime.\n");
            unload();
            return false;
        }

        int status = network_->network_create(const_cast<char*>(model_path.c_str()), network_id_);
        if (status != 0) {
            printf("awnn_inference: failed to create network from %s.\n", model_path.c_str());
            unload();
            return false;
        }

        status = network_->network_prepare();
        if (status != 0) {
            printf("awnn_inference: failed to prepare network.\n");
            unload();
            return false;
        }

        input_count_ = probe_input_count();
        output_count_ = network_->get_output_cnt();
        loaded_ = true;
        return true;
    }

    void unload()
    {
        outputs_cache_.clear();
        output_info_cache_.clear();
        output_count_ = 0;
        input_count_ = 0;
        loaded_ = false;
        network_.reset();
        npu_.reset();
    }

    bool is_loaded() const
    {
        return loaded_;
    }

    int input_count() const
    {
        return input_count_;
    }

    int output_count() const
    {
        return output_count_;
    }

    bool input_buffer(int index, BufferView* view)
    {
        if (!loaded_ || view == nullptr) {
            return false;
        }

        void* ptr = nullptr;
        unsigned int size = 0;
        if (network_->get_network_input_buff_info(index, &ptr, &size) != 0) {
            return false;
        }

        view->data = ptr;
        view->bytes = size;
        return true;
    }

    bool copy_input(int index, const void* data, unsigned int bytes)
    {
        if (!loaded_ || data == nullptr) {
            return false;
        }

        return network_->network_load_input_buffer_idx(const_cast<void*>(data), bytes, index) == 0;
    }

    bool run(RunProfile* profile)
    {
        if (!loaded_) {
            return false;
        }

        timespec total_begin {};
        timespec total_end {};
        timespec bind_begin {};
        timespec bind_end {};
        timespec run_begin {};
        timespec run_end {};
        clock_gettime(CLOCK_MONOTONIC, &total_begin);

        clock_gettime(CLOCK_MONOTONIC, &bind_begin);
        int status = network_->network_input_output_set();
        clock_gettime(CLOCK_MONOTONIC, &bind_end);
        if (status != 0) {
            printf("awnn_inference: failed to bind network input/output, status=%d.\n", status);
            return false;
        }

        clock_gettime(CLOCK_MONOTONIC, &run_begin);
        status = network_->network_run();
        clock_gettime(CLOCK_MONOTONIC, &run_end);
        if (status != 0) {
            printf("awnn_inference: failed to run network, status=%d.\n", status);
            return false;
        }

        clock_gettime(CLOCK_MONOTONIC, &total_end);

        if (profile != nullptr) {
            profile->bind_ms = elapsed_ms(bind_begin, bind_end);
            profile->run_ms = elapsed_ms(run_begin, run_end);
            profile->total_ms = elapsed_ms(total_begin, total_end);
        }

        return true;
    }

    bool output_tensors(std::vector<TensorView>* outputs)
    {
        if (!loaded_ || outputs == nullptr) {
            return false;
        }

        output_info_cache_.assign(output_count_, output_info_s{});
        network_->get_output_fp_nocopy(output_info_cache_.data());

        outputs_cache_.clear();
        outputs_cache_.reserve(output_count_);
        for (int i = 0; i < output_count_; ++i) {
            TensorView view;
            view.data = output_info_cache_[i].ptr;
            view.elements = static_cast<unsigned int>(output_info_cache_[i].length);
            view.index = i;
            outputs_cache_.push_back(view);
        }

        *outputs = outputs_cache_;
        return true;
    }

    bool output_tensor(int index, TensorView* output)
    {
        if (output == nullptr) {
            return false;
        }

        std::vector<TensorView> outputs;
        if (!output_tensors(&outputs)) {
            return false;
        }

        if (index < 0 || index >= static_cast<int>(outputs.size())) {
            return false;
        }

        *output = outputs[index];
        return true;
    }

private:
    int probe_input_count()
    {
        void* ptr = nullptr;
        unsigned int size = 0;
        return network_->get_network_input_buff_info(0, &ptr, &size) == 0 ? 1 : 0;
    }

    bool loaded_ = false;
    int input_count_ = 0;
    int output_count_ = 0;
    unsigned int network_id_ = 0;
    std::unique_ptr<NpuUint> npu_;
    std::unique_ptr<NetworkItem> network_;
    std::vector<output_info_s> output_info_cache_;
    std::vector<TensorView> outputs_cache_;
};

Session::Session() : impl_(new Impl()) {}

Session::~Session()
{
    delete impl_;
    impl_ = nullptr;
}

bool Session::load(const std::string& model_path)
{
    return impl_->load(model_path);
}

void Session::unload()
{
    impl_->unload();
}

bool Session::is_loaded() const
{
    return impl_->is_loaded();
}

int Session::input_count() const
{
    return impl_->input_count();
}

int Session::output_count() const
{
    return impl_->output_count();
}

bool Session::input_buffer(int index, BufferView* view)
{
    return impl_->input_buffer(index, view);
}

bool Session::copy_input(int index, const void* data, unsigned int bytes)
{
    return impl_->copy_input(index, data, bytes);
}

bool Session::run(RunProfile* profile)
{
    return impl_->run(profile);
}

bool Session::output_tensors(std::vector<TensorView>* outputs)
{
    return impl_->output_tensors(outputs);
}

bool Session::output_tensor(int index, TensorView* output)
{
    return impl_->output_tensor(index, output);
}

}  // namespace awnn
