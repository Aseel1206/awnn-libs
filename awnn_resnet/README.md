# awnn_resnet

`awnn_resnet` is a classification postprocess library for ResNet-v2-style models.

It sits above `awnn_inference` and turns raw tensors into:

- top-k class predictions
- class ids
- scores

It supports both:

- normal multi-class heads like ImageNet `1000`-class output
- binary single-output heads such as `mine / not_mine`

It can also render the top prediction text:

- onto a copied image
- or in-place on the same image buffer

## Intended stack

```text
image path
-> awnn_vision or custom preprocess
-> awnn_inference
-> awnn_resnet
-> ROS publishes class scores / geotagged metadata
```

## Public API

Header:

```cpp
#include "awnn_resnet/decoder.h"
```

Main objects:

- `RawTensorView`
- `ResnetModelConfig`
- `ResnetPrediction`
- `ResnetDecodeResult`
- `ResnetDecoder`

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
- `awnn_resnet` for decode and optional overlay

Run:

```bash
cd install
export LD_LIBRARY_PATH=$PWD/lib:${LD_LIBRARY_PATH}
./awnn_resnet_pipeline_demo -m /path/to/resnetv2.nb -i /path/to/image.jpg --overlay copy -o resnet_overlay.png
```

Overlay modes:

- `--overlay none`: decode only, no image written
- `--overlay copy`: draw on a copy and save it
- `--overlay inplace`: draw on the same in-memory image and save that result

## Notes

- This decoder expects a single FP32 output tensor containing class scores.
- The pipeline demo auto-detects a single scalar output and treats it as binary classification with labels:
  - `0 = not_mine`
  - `1 = mine`
- For binary output, if the scalar is already in `[0, 1]`, it is treated as a probability.
- If the scalar falls outside `[0, 1]`, the decoder applies sigmoid automatically before scoring.
- For non-binary output, the default demo configuration assumes ImageNet-style `1000` classes and `224x224` input.
- The bundled multi-class demo labels are placeholders like `class_0 ... class_999`.
