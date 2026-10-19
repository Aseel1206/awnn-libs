# awnn_cls

`awnn_cls` is a classification postprocess library for MobileNetV2-style models.

It sits above `awnn_inference` and turns raw tensors into:

- top-k class predictions
- class ids
- scores

It can also render the top prediction text:

- onto a copied image
- or in-place on the same image buffer

## Intended stack

```text
image path
-> awnn_vision or custom preprocess
-> awnn_inference
-> awnn_cls
-> ROS publishes class scores / geotagged metadata
```

## Public API

Header:

```cpp
#include "awnn_cls/decoder.h"
```

Main objects:

- `RawTensorView`
- `ClsModelConfig`
- `ClsPrediction`
- `ClsDecodeResult`
- `ClsDecoder`

## Build on target

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

chmod +x build_on_target.sh
./build_on_target.sh
```

## Complete flow demo

This folder includes a standalone end-to-end demo that uses:

- `awnn_vision` for preprocessing
- `awnn_inference` for NPU execution
- `awnn_cls` for decode and optional overlay

Run:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_cls_pipeline_demo -m /path/to/model.nb -i /path/to/image.jpg --overlay copy -o cls_overlay.png
```

Overlay modes:

- `--overlay none`: decode only, no image written
- `--overlay copy`: draw on a copy and save it
- `--overlay inplace`: draw on the same in-memory image and save that result

## Notes

- This decoder expects a single FP32 output tensor containing class scores.
- Default demo configuration assumes ImageNet-style `1000` classes and `224x224` input.
