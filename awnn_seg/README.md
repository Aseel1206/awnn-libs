# awnn_seg

`awnn_seg` is a segmentation postprocess library that sits above `awnn_inference`.

It takes raw model tensors and returns structured segmentation results:

- bounding boxes
- class ids
- scores
- binary masks

It can also render masks and boxes:

- onto a copied image
- or directly in-place on the original image buffer

## Intended stack

```text
image path
-> awnn_vision
-> awnn_inference
-> awnn_seg
-> ROS publishes detections / masks / geotagged results
```

## Public API

Header:

```cpp
#include "awnn_seg/decoder.h"
```

Core objects:

- `SegModelConfig`
- `RawTensorView`
- `SegDetection`
- `SegDecodeResult`
- `SegDecoder`

### API reference

Header:

```cpp
#include "awnn_seg/decoder.h"
```

`awnn::seg::RawTensorView`

- `data`: pointer to raw FP32 tensor
- `elements`: number of float elements
- `index`: output tensor index

`awnn::seg::SegModelConfig`

- `input_width`, `input_height`: model input size
- `class_count`: number of classes
- `score_threshold`: confidence threshold
- `nms_threshold`: NMS threshold
- `mask_threshold`: binary mask threshold
- `mask_proto_channels`: proto channel count, currently `32`
- `mask_stride`: proto stride, currently `4`
- `class_names`: label names used for overlay text

`awnn::seg::SegDetection`

- `box`: `cv::Rect2f` in original image coordinates
- `class_id`: predicted class id
- `score`: confidence score
- `mask`: binary `cv::Mat` mask for the detection ROI

`awnn::seg::SegDecodeResult`

- `detections`: vector of decoded detections
- `decode_ms`: decode timing

`awnn::seg::SegDecoder`

- `explicit SegDecoder(SegModelConfig config = {})`
- `bool decode(const cv::Mat& bgr_image, const std::vector<RawTensorView>& outputs, SegDecodeResult* result) const`
- `bool render_overlay(const cv::Mat& input_image, const SegDecodeResult& result, cv::Mat* output_image) const`
- `bool render_overlay_in_place(cv::Mat* image, const SegDecodeResult& result) const`
- `const SegModelConfig& config() const`

## ROS access

This library is the postprocess step your ROS node calls after `awnn_inference`.

Typical ROS-side flow:

```cpp
#include "awnn_seg/decoder.h"

awnn::seg::SegModelConfig config;
config.input_width = 640;
config.input_height = 640;
config.class_count = 1;
config.class_names = {"mine"};

awnn::seg::SegDecoder decoder(config);

std::vector<awnn::seg::RawTensorView> seg_outputs;
seg_outputs.reserve(outputs.size());
for (const auto& output : outputs) {
    awnn::seg::RawTensorView view;
    view.data = output.data;
    view.elements = output.elements;
    view.index = output.index;
    seg_outputs.push_back(view);
}

awnn::seg::SegDecodeResult result;
decoder.decode(original_bgr_image, seg_outputs, &result);
```

After decode, ROS can use:

- `result.detections[i].box` for geotag association
- `result.detections[i].mask` for segmented region processing
- `result.detections[i].score`
- `result.detections[i].class_id`

### Overlay behavior for ROS

- `render_overlay(...)`: keeps the original `cv::Mat` unchanged and writes to a separate output image
- `render_overlay_in_place(...)`: modifies the same `cv::Mat` in memory

Important: neither function overwrites the original image file on disk unless your node explicitly saves that image back to the same file path.

## Build on target

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

chmod +x build_on_target.sh
./build_on_target.sh
```

## Overlay behavior

Use:

- `render_overlay(...)` to render onto a copy
- `render_overlay_in_place(...)` to modify the same input image buffer

That gives your ROS node the choice at runtime.

Important: `render_overlay_in_place(...)` only modifies the `cv::Mat` in memory. It does not overwrite the original image file unless your code explicitly saves that image back to disk with `cv::imwrite(...)`.

## Complete flow demo

This folder also includes a standalone end-to-end demo that uses:

- `awnn_vision` for preprocessing
- `awnn_inference` for NPU execution
- `awnn_seg` for decode and optional overlay

Build:

```bash
chmod +x build_on_target.sh
./build_on_target.sh
```

Run:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_seg_pipeline_demo -m /path/to/model.nb -i /path/to/image.jpg --overlay copy -o overlay.png
```

Overlay modes:

- `--overlay none`: decode only, no image written
- `--overlay copy`: draw on a copy and save it
- `--overlay inplace`: draw on the same in-memory image and save that result

## End-to-end ROS-style flow

In practice, your C++ ROS node will look like:

1. Choose image path from folder/geotag workflow.
2. Call `awnn_inference::Session::input_buffer(...)`.
3. Call `awnn_vision::fill_rgb_u8_letterbox_from_path(...)`.
4. Call `session.run(...)`.
5. Call `session.output_tensors(...)`.
6. Convert outputs to `awnn::seg::RawTensorView`.
7. Call `decoder.decode(...)`.
8. Publish boxes, masks, scores, class ids, and geotagged metadata.
9. Optionally call `render_overlay(...)` or `render_overlay_in_place(...)`.
