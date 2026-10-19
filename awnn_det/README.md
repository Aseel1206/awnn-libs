# awnn_det

`awnn_det` is an object detection postprocess library for YOLO26-style 6-output models.

It sits above `awnn_inference` and turns raw tensors into:

- bounding boxes
- class ids
- scores

It can also render detections:

- onto a copied image
- or in-place on the same image buffer

## Intended stack

```text
image path
-> awnn_vision
-> awnn_inference
-> awnn_det
-> ROS publishes boxes/classes/scores/geotagged metadata
```

## Public API

Header:

```cpp
#include "awnn_det/decoder.h"
```

Main objects:

- `DetModelConfig`
- `RawTensorView`
- `DetDetection`
- `DetDecodeResult`
- `DetDecoder`

## Build on target

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

chmod +x build_on_target.sh
./build_on_target.sh
```

## Complete flow demo

This folder also includes a standalone end-to-end demo that uses:

- `awnn_vision` for preprocessing
- `awnn_inference` for NPU execution
- `awnn_det` for decode and optional overlay

Run:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_det_pipeline_demo -m /path/to/model.nb -i /path/to/image.jpg --overlay copy -o det_overlay.png
```

Overlay modes:

- `--overlay none`: decode only, no image written
- `--overlay copy`: draw on a copy and save it
- `--overlay inplace`: draw on the same in-memory image and save that result

## ROS usage

The expected ROS flow is:

1. preprocess image with `awnn_vision`
2. run model with `awnn_inference`
3. convert `awnn::TensorView` to `awnn::det::RawTensorView`
4. call `DetDecoder::decode(...)`
5. optionally render with `render_overlay(...)` or `render_overlay_in_place(...)`

## Notes

- This decoder is designed for the pretrained YOLO26 object-detection output layout with 6 FP32 outputs.
- It is model-specific at the decode level, while `awnn_inference` remains generic.
