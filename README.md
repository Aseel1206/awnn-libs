# awnn libs

A set of small, ROS-independent C++ libraries for running neural-network models on the
NPU of Allwinner SoCs (built against the A733 vendor runtime from the
[AWNPU Model Zoo](https://open.allwinnertech.com/)).

The libraries split a typical inference pipeline into reusable layers:

```text
image path / cv::Mat
  -> awnn_vision      preprocess (resize + letterbox -> RGB uint8 HWC tensor)
  -> awnn_inference   load .nb model, run on NPU, return raw FP32 output tensors
  -> awnn_cls | awnn_resnet | awnn_det | awnn_seg
                      model-specific postprocess (top-k, boxes + NMS, masks) and overlay drawing
```

## Libraries

| Library | Role | Depends on |
| --- | --- | --- |
| [`awnn_inference`](awnn_inference) | Low-level NPU session: load `.nb` model, fill input, run, read FP32 outputs. Bundles the vendor runtime (`libVIPhal.so`, `libNBGlinker.so`, wrapper sources). | nothing (self-contained) |
| [`awnn_vision`](awnn_vision) | OpenCV preprocessing: load image, letterbox to model size, write RGB/uint8/HWC bytes straight into the NPU input buffer. | OpenCV |
| [`awnn_cls`](awnn_cls) | Top-k decoder for MobileNetV2-style classifiers (single FP32 score tensor). | OpenCV, `awnn_inference`, `awnn_vision` (demo) |
| [`awnn_resnet`](awnn_resnet) | Top-k decoder for ResNet-v2-style classifiers; also handles binary single-output heads (auto sigmoid). | OpenCV, `awnn_inference`, `awnn_vision` (demo) |
| [`awnn_det`](awnn_det) | YOLO26-style object-detection decoder (6 FP32 outputs): boxes, class ids, scores, NMS. | OpenCV, `awnn_inference`, `awnn_vision` (demo) |
| [`awnn_seg`](awnn_seg) | YOLO-style instance-segmentation decoder: boxes, class ids, scores, binary masks. | OpenCV, `awnn_inference`, `awnn_vision` (demo) |

Each library has its own `README.md` with its full API reference.

## Requirements

These libraries are built **on the target board** (Linux, aarch64) that has the NPU kernel driver:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

- CMake >= 3.16, a C++14 compiler
- OpenCV 4 discoverable by `pkg-config` (`opencv4`) - needed by everything except `awnn_inference`
- A model converted to the Allwinner NPU `.nb` format (see the parent Model Zoo's
  `scripts_model_convert/` and `examples/` for conversion and quantization steps)

## Repository layout

```text
libs/
  awnn_inference/   include/  src/  vendor/{include,src,lib}  examples/smoketest.cpp
  awnn_vision/      include/  src/  examples/smoketest.cpp
  awnn_cls/         include/  src/  examples/pipeline_demo.cpp
  awnn_resnet/      include/  src/  examples/pipeline_demo.cpp
  awnn_det/         include/  src/  examples/{pipeline_demo,batch_pipeline_demo}.cpp
  awnn_seg/         include/  src/  examples/pipeline_demo.cpp
  <each>/CMakeLists.txt, build_on_target.sh
```

The decoder libraries (`cls`, `resnet`, `det`, `seg`) pull in `awnn_inference` and `awnn_vision`
via `add_subdirectory(../awnn_inference)` / `../awnn_vision`, so **keep the folders side by side**
as in this repo. `awnn_inference` is the only folder that can be copied to a board on its own.

## Build

Every library has a `build_on_target.sh` that configures, builds and installs into `<lib>/install`:

```bash
cd awnn_det            # or awnn_inference, awnn_vision, awnn_cls, awnn_resnet, awnn_seg
chmod +x build_on_target.sh
./build_on_target.sh
export LD_LIBRARY_PATH=$PWD/install/lib:${LD_LIBRARY_PATH}
```

Install output (example for `awnn_det`):

```text
install/include/awnn_det/decoder.h
install/lib/libawnn_det.so  (+ libawnn_inference.so, libawnn_vision.so, libVIPhal.so, libNBGlinker.so)
install/awnn_det_pipeline_demo
install/awnn_det_batch_pipeline_demo
```

## Quick start

### 1. Verify the NPU runtime

```bash
cd awnn_inference && ./build_on_target.sh
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}

./awnn_inference_smoketest -m /path/to/model.nb -z                 # load + one dummy run
./awnn_inference_smoketest -m /path/to/model.nb -i /path/to/input.bin   # run a raw input tensor
```

### 2. Run a full image -> result demo

Each decoder ships an end-to-end demo (preprocess -> NPU -> decode -> optional overlay):

```bash
cd awnn_det && ./build_on_target.sh && cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}

./awnn_det_pipeline_demo -m /path/to/model.nb -i /path/to/image.jpg --overlay copy -o det_overlay.png
```

| Library | Demo binary | Typical arguments |
| --- | --- | --- |
| `awnn_cls` | `awnn_cls_pipeline_demo` | `-m model.nb -i image.jpg --overlay copy -o out.png` |
| `awnn_resnet` | `awnn_resnet_pipeline_demo` | `-m model.nb -i image.jpg --overlay copy -o out.png` |
| `awnn_det` | `awnn_det_pipeline_demo` | `-m model.nb -i image.jpg --overlay copy -o out.png` |
| `awnn_det` | `awnn_det_batch_pipeline_demo` | `-m model.nb -i <input_dir> -o <output_dir> --overlay copy` |
| `awnn_seg` | `awnn_seg_pipeline_demo` | `-m model.nb -i image.jpg --overlay copy -o out.png` |

`--overlay` modes: `none` (decode only), `copy` (draw on a copy and save), `inplace` (draw on the
in-memory image and save it). Neither mode overwrites the source file unless you save to that path.

## Using the libraries in your own code

Linking is plain CMake - add the folders you need as subdirectories:

```cmake
add_subdirectory(libs/awnn_inference)
add_subdirectory(libs/awnn_vision)
add_subdirectory(libs/awnn_det)

target_link_libraries(my_app PRIVATE awnn_det awnn_inference awnn_vision)
```

A complete detection pipeline:

```cpp
#include "awnn_inference/session.h"
#include "awnn_vision/image_preprocessor.h"
#include "awnn_det/decoder.h"

// 1. Load the model once at startup.
awnn::Session session;
if (!session.load("model.nb")) { /* handle error */ }

// 2. Preprocess straight into the NPU input buffer.
cv::Mat bgr;
awnn::vision::load_bgr_image("image.jpg", &bgr);

awnn::BufferView input;
session.input_buffer(0, &input);
awnn::vision::fill_rgb_u8_letterbox_from_mat(bgr, 640, 640, input.data, input.bytes);

// 3. Run on the NPU and fetch raw FP32 tensors.
awnn::RunProfile profile;
session.run(&profile);
std::vector<awnn::TensorView> outputs;
session.output_tensors(&outputs);

// 4. Wrap tensors for the decoder and decode.
std::vector<awnn::det::RawTensorView> raw;
for (const auto& o : outputs) raw.push_back({o.data, o.elements, o.index});

awnn::det::DetModelConfig cfg;          // 640x640, 80 classes, score 0.4, NMS 0.45
awnn::det::DetDecoder decoder(cfg);

awnn::det::DetDecodeResult result;
decoder.decode(bgr, raw, &result);      // boxes are in original-image coordinates

decoder.render_overlay_in_place(&bgr, result);
cv::imwrite("out.png", bgr);
```

The other decoders follow the same shape - only steps 4+ change:

- **Classification** (`awnn_cls`, `awnn_resnet`): `decoder.decode(raw, &result)` returns
  `predictions` (class id, score, label); `awnn_resnet` supports `binary_single_output`.
- **Segmentation** (`awnn_seg`): `decoder.decode(bgr, raw, &result)` additionally gives a binary
  `mask` per detection; configure `class_count`, `class_names`, `mask_threshold`.

Notes:

- `TensorView::data` points at vendor-managed memory and is valid only until the next `run()` or `unload()`. Copy results you need to keep.
- `Session` is non-copyable; create one per model and reuse it across frames.
- The decoders are model-specific (output layout, class count, input size); `awnn_inference` itself is generic.
- The libraries are C++ only. From Python, add a C wrapper (`ctypes`/`cffi`) or `pybind11` bindings.

## Notes and limitations

- `awnn_inference/vendor/lib/` holds **A733** runtime binaries; other SoCs (V853, T527, T536) need their own runtime libs.
- `awnn_det` is written for the YOLO26 6-output layout; `awnn_seg` expects 32 mask-proto channels at stride 4 (YOLOv8/11-seg style).
- Demo class names are placeholders (e.g. `mine` / `not mine`, `class_0 ... class_999`) - set `class_names` in each config.

## License / attribution

The code under `awnn_inference/vendor/` is Allwinner / Vivante Corporation vendor code
(Vivante portions are MIT-licensed, see headers in `vendor/include`). The prebuilt `.so` files
are redistributed from the Allwinner AWNPU Model Zoo; confirm you are permitted to redistribute
them before making this repository public.
