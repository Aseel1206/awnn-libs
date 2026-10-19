# awnn_vision

`awnn_vision` is a small ROS-independent preprocessing library for turning images into model input tensors.

It is intended to sit above `awnn_inference`:

```text
image path / cv::Mat
-> awnn_vision
-> input tensor bytes
-> awnn_inference
-> raw model outputs
```

This library uses system OpenCV on the board via `pkg-config`.

## Public API

Header:

```cpp
#include "awnn_vision/image_preprocessor.h"
```

Main helpers:

- `load_bgr_image(path, &image)`
- `fill_rgb_u8_letterbox_from_path(...)`
- `fill_rgb_u8_letterbox_from_mat(...)`

### API reference

Header:

```cpp
#include "awnn_vision/image_preprocessor.h"
```

`awnn::vision::LetterboxInfo`

- `target_width`, `target_height`: model input size
- `resized_width`, `resized_height`: resized image before padding
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom`: letterbox padding
- `scale`: resize scale used

`awnn::vision::PreprocessProfile`

- `imread_ms`: image file read time
- `preprocess_ms`: resize/color/letterbox time
- `total_ms`: full preprocessing time

`bool load_bgr_image(const std::string& image_path, cv::Mat* image)`

- `image_path`: file path to image on disk
- `image`: output `cv::Mat` in BGR layout

`bool fill_rgb_u8_letterbox_from_path(...)`

- `image_path`: file path to image on disk
- `target_width`, `target_height`: model input size, for your seg model typically `640, 640`
- `output_buffer`: pointer to model input tensor memory
- `output_bytes`: size of output buffer in bytes
- `letterbox`: optional metadata output
- `profile`: optional timing output

`bool fill_rgb_u8_letterbox_from_mat(...)`

- same as above, but takes an already-loaded `cv::Mat`

This helper currently produces:

- `RGB`
- `uint8`
- `HWC`
- letterboxed tensor bytes

## ROS access

For your folder-based workflow, the ROS node can call this library directly with the image path selected from geotag/topic logic.

Typical flow:

```cpp
#include "awnn_inference/session.h"
#include "awnn_vision/image_preprocessor.h"

awnn::BufferView input;
session.input_buffer(0, &input);

awnn::vision::LetterboxInfo letterbox;
awnn::vision::PreprocessProfile prep_profile;
awnn::vision::fill_rgb_u8_letterbox_from_path(
    image_path,
    640,
    640,
    input.data,
    input.bytes,
    &letterbox,
    &prep_profile);
```

This means ROS passes image paths or `cv::Mat`s, not `.bin` files.

## Build on target

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

chmod +x build_on_target.sh
./build_on_target.sh
```

## Smoke test

```bash
cd install
./awnn_vision_smoketest -i /path/to/image.jpg -w 640 -h 640
```

## Intended ROS usage

- ROS receives image path from folder workflow
- node calls `awnn_vision` to fill tensor bytes
- node passes that tensor to `awnn_inference`
- node does model-specific postprocess itself

## Python access

Right now `awnn_vision` and `awnn_inference` expose C++ APIs, so they are easiest to use from C++ ROS nodes.

Python can access them later, but not directly as a nice Python API yet. For Python, the clean options are:

- add a C wrapper and use `ctypes`/`cffi`
- add `pybind11` bindings
- keep the ROS node in C++ and publish results to Python nodes

For your current stack, a C++ ROS node is the simplest and most reliable choice.
