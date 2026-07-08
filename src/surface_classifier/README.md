# surface_classifier

ROS 2 package that classifies the ground patch in front of an F1Tenth car
(RealSense RGB, ~20 cm above the ground) into one of 4 **materials**
(`carpet`, `cloth`, `sticky`, `tile`) and publishes both the material and
a derived **friction tier** (`low` / `medium` / `high`). The material-to-tier
mapping is data-driven via `config/friction_map.yaml`, which also covers
additional materials (`ice`, `grass`, `road`) for future retraining.

Designed for the NVIDIA Jetson Orin Nano Super. Inference uses a TensorRT FP16
engine, mirroring the conventions used by the existing `real_sense` detection
package.

---

## End-to-end workflow

```
[1] collect labeled patches  →  recorder_node                              (on car)
[2] split + train             →  scripts/make_split.py + scripts/train.py   (on Colab/laptop)
[3] export to ONNX            →  scripts/export_onnx.py                     (on Colab/laptop)
[4] build TRT engine          →  scripts/convert_trt.py                     (on Jetson)
[5] live inference            →  classifier_node                            (on car)
```

---

## 1. Build the package

```bash
cd <ros2_ws>
colcon build --packages-select surface_classifier --symlink-install
source install/setup.bash
```

The package depends on `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, plus
`pyyaml`, `opencv-python`, `numpy`, `pycuda`, and `tensorrt` (the last two only
needed at inference time on the Jetson).

---

## 2. Collect data (recorder_node)

Start the existing camera node in another terminal:

```bash
ros2 run real_sense camera_node
```

The `real_sense` camera node publishes `/camera/rgb/raw` at **640×480 BGR8, 30 Hz**
(see `lab-7-vision-lab-team3/real_sense/scripts/camera_node.py`). The default ROI
in `config/classifier.yaml` is sized for that resolution. If you switch to a
different camera publisher (e.g., the standalone 960×540 pipeline used in
`lab-7-vision-lab-team3/vision_integration_pkg/scripts/integrated.py`), retune
`roi_x/y/w/h` accordingly.

Then, on the car:

```bash
ros2 launch surface_classifier recorder.launch.py current_label:=tile
```

Drive over the surface for ~3–5 minutes (varied lighting / orientation /
exact location). Patches save to `~/surface_data/tile/<unix_ts>.png` at 2 Hz.

To switch label without restarting:

```bash
ros2 param set /recorder_node current_label grass
```

**Recommended target:** ~200 patches/class for v1.

For `ice`, use a low-friction proxy (acrylic sheet, polished laminate, wet
tile) — actual ice is impractical and unsafe for the car.

---

## 3. Train (offline, Colab T4 or laptop GPU)

Copy `~/surface_data` to your training machine, then:

```bash
python scripts/make_split.py --root ~/surface_data --out-dir . --val-frac 0.2
python scripts/train.py \
    --train-manifest manifest_train.csv \
    --val-manifest manifest_val.csv \
    --classes classes.txt \
    --out models/surface_mnv3.pt \
    --epochs 30 --batch-size 32
```

Expected runtime: ~5–15 min on Colab T4 for ~1000 total patches, 30 epochs.
Target: val accuracy 85–92% on in-distribution surfaces.

---

## 4. Export ONNX → build TRT engine

On the training machine:

```bash
python scripts/export_onnx.py --ckpt models/surface_mnv3.pt --out models/surface_mnv3.onnx
```

Copy `models/surface_mnv3.onnx` to the Jetson, then build the TensorRT engine
using either of the following (both produce the same `.trt` file):

```bash
# Preferred — Python API, mirrors lab-7 python_scripts/convert_trt.py
python scripts/convert_trt.py --onnx models/surface_mnv3.onnx \
    --engine models/surface_mnv3_fp16.trt --fp16

# Alternative — uses NVIDIA's trtexec CLI (one-liner, identical output)
./scripts/build_trt_engine.sh models/surface_mnv3.onnx models/surface_mnv3_fp16.trt
```

Place the engine at the path referenced by `engine_path` in
`config/classifier.yaml` (default: `/opt/surface_classifier/models/surface_mnv3_fp16.trt`).

Update `class_names` in `config/classifier.yaml` to match the order in
`classes.txt` (alphabetical from `make_split.py`).

---

## 5. Run live inference (classifier_node)

```bash
ros2 launch surface_classifier classifier.launch.py   # run this 1st
ros2 run real_sense camera_node                 # run this 2nd
```

### Topics published

| Topic                  | Type                  | Notes                              |
| ---------------------- | --------------------- | ---------------------------------- |
| `/surface/material`    | `std_msgs/String`     | smoothed material class            |
| `/surface/friction`    | `std_msgs/String`     | `low` / `medium` / `high`          |
| `/surface/confidence`  | `std_msgs/Float32`    | max softmax of latest raw prediction |
| `/surface/debug_image` | `sensor_msgs/Image`   | ROI crop with label overlay (RViz) |

### Inspect

```bash
ros2 topic echo /surface/material
ros2 topic echo /surface/friction
ros2 topic hz /surface/material        # expect ~10 Hz
rviz2                                  # add Image -> /surface/debug_image
```

---

## Configuration

All runtime parameters live in `config/classifier.yaml`:

- **ROI** (`roi_x/y/w/h`): the bottom-center crop on the 640×480 BGR frame.
  Default `(220, 320, 200, 160)` — tune by inspecting `/surface/debug_image`.
- **Inference rate** (`inference_hz`): default 10 Hz, well below the 30 Hz
  camera rate.
- **Smoothing** (`smoothing_window`): majority-vote ring buffer size, default 5.
- **Friction map** (`friction_map_path` → `config/friction_map.yaml`): edit
  freely without retraining.

---

## Files

```
surface_classifier/
├── package.xml
├── setup.py / setup.cfg
├── resource/surface_classifier
├── surface_classifier/
│   ├── __init__.py
│   ├── classifier_node.py     # inference ROS node
│   ├── recorder_node.py       # data-collection ROS node
│   ├── trt_classifier.py      # TensorRT wrapper
│   └── friction_map.py
├── config/
│   ├── classifier.yaml
│   └── friction_map.yaml
├── launch/
│   ├── classifier.launch.py
│   └── recorder.launch.py
├── scripts/
│   ├── make_split.py
│   ├── train.py
│   ├── export_onnx.py
│   ├── convert_trt.py         # primary engine builder (Python API)
│   └── build_trt_engine.sh    # alternative engine builder (trtexec)
└── models/                    # populated after step 4
```

---

## Verification checklist

1. Recorder smoke test: 30 s drive over tile → ~60 PNGs in `~/surface_data/tile/`.
2. Training sanity: 3 epochs on 50 patches/class → val acc >> 20% (random).
3. TRT engine built and loadable: `python -c "import tensorrt as trt; r=trt.Runtime(trt.Logger()); e=r.deserialize_cuda_engine(open('models/surface_mnv3_fp16.trt','rb').read()); print('ok', e.num_io_tensors)"`. Optionally benchmark with `trtexec --loadEngine=models/surface_mnv3_fp16.trt --useCudaGraph` (< 5 ms expected on Orin Nano).
4. Live inference: hold car over each surface → correct material on `/surface/material`.
5. Throughput: `ros2 topic hz /surface/material` ≈ 10 Hz; end-to-end < 50 ms.
6. Confusion matrix on a held-out drive: off-diagonal errors should not cross
   friction tiers (carpet↔grass OK; tile↔road is a real failure).
