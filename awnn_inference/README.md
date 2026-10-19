# awnn_inference

`awnn_inference` is a small C++ library that wraps the Allwinner vendor NPU runtime behind a stable, ROS-independent API.

This folder is self-contained, so you can copy only `libs/awnn_inference` to the board and build it there.

It is intentionally low-level:

- model load and unload
- access to input buffers
- run inference
- retrieve raw FP32 output tensors

It does not include:

- OpenCV preprocessing
- model-specific postprocessing
- ROS code

## Bundled vendor runtime

This folder already contains the vendor-side pieces it needs:

- `vendor/src/`: copied Allwinner wrapper sources
- `vendor/include/`: copied vendor headers
- `vendor/lib/`: copied A733 runtime shared libraries

That means the dependency chain is fully inside this folder:

```text
your app or ROS node
-> libawnn_inference.so
-> bundled vendor wrapper sources
-> bundled libVIPhal.so / libNBGlinker.so
-> board kernel NPU driver
```

## Public API

Header:

```cpp
#include "awnn_inference/session.h"
```

Core usage:

```cpp
awnn::Session session;
session.load("model.nb");

awnn::BufferView input;
session.input_buffer(0, &input);
// Fill input.data with your preprocessed bytes here.

awnn::RunProfile profile;
session.run(&profile);

std::vector<awnn::TensorView> outputs;
session.output_tensors(&outputs);
```

`outputs[i].data` points at vendor-managed output memory and stays valid until the next `run()` or `unload()`.

### API reference

`awnn::BufferView`

- `data`: writable pointer to the vendor-managed input buffer
- `bytes`: input buffer size in bytes

`awnn::TensorView`

- `data`: pointer to FP32 output tensor memory
- `elements`: number of float elements
- `index`: output tensor index

`awnn::RunProfile`

- `bind_ms`: time to bind input/output buffers
- `run_ms`: pure NPU execution time
- `total_ms`: bind + run timing measured inside `run()`

`awnn::Session`

- `bool load(const std::string& model_path)`
  - `model_path`: absolute or relative path to `.nb` model file
- `void unload()`
- `bool is_loaded() const`
- `int input_count() const`
- `int output_count() const`
- `bool input_buffer(int index, BufferView* view)`
  - `index`: model input index, usually `0` for your current seg model
  - `view`: filled with input pointer and size
- `bool copy_input(int index, const void* data, unsigned int bytes)`
  - copies preprocessed bytes into input buffer
- `bool run(RunProfile* profile = nullptr)`
  - `profile`: optional timing output
- `bool output_tensors(std::vector<TensorView>* outputs)`
- `bool output_tensor(int index, TensorView* output)`

## ROS access

This library is meant to be called from a C++ ROS node, not through ROS messages directly.

Typical ROS-side flow:

1. Create one `awnn::Session` in your node class.
2. Call `load(model_path)` once during startup.
3. For each image:
   - get input buffer with `input_buffer(0, &view)`
   - fill it using `awnn_vision`
   - call `run(&profile)`
   - fetch raw tensors with `output_tensors(&outputs)`
4. Pass raw tensors to `awnn_seg` or your own postprocess code.
5. Call `unload()` during shutdown if needed.

Minimal C++ example:

```cpp
#include "awnn_inference/session.h"

awnn::Session session;
if (!session.load(model_path)) {
    throw std::runtime_error("failed to load model");
}

awnn::BufferView input;
session.input_buffer(0, &input);

// Fill input.data here.

awnn::RunProfile profile;
session.run(&profile);

std::vector<awnn::TensorView> outputs;
session.output_tensors(&outputs);
```

## Build on target

```bash
chmod +x build_on_target.sh
./build_on_target.sh
```

Then on the board, if needed:

```bash
export LD_LIBRARY_PATH=$PWD/install/lib:${LD_LIBRARY_PATH}
```

## Install output

- `install/include/awnn_inference/session.h`
- `install/lib/libawnn_inference.so`
- `install/lib/libVIPhal.so`
- `install/lib/libNBGlinker.so`
- `install/awnn_inference_smoketest`

## Quick smoke test

If you already have a raw input tensor file:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_inference_smoketest -m /path/to/model.nb -i /path/to/input.bin
```

If you just want to verify model load plus one dummy run:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_inference_smoketest -m /path/to/model.nb -z
```

The `-z` mode is only a runtime smoke test. It does not produce meaningful detections.

## Intended ROS usage

This library should sit below your ROS nodes:

- ROS node subscribes to image topics
- ROS node preprocesses images or calls your own preprocess helper
- ROS node calls `awnn::Session`
- ROS node does postprocess and publishes detections/masks
