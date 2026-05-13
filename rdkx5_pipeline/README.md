# RDK X5 anomaly prefilter pipeline

This version avoids command-line arguments. Open each numbered script, change the
small config block at the top, then run the file.

## 01_train_float_stats.py

PC-side prototype. It uses a PyTorch backbone to extract normal-image features
and saves `float_anomaly_stats.npz`.

Use this only to check whether the chosen backbone is useful for your data.

## 02_export_feature_onnx.py

Exports the feature extractor to ONNX.

The ONNX output is `feature`, not the final anomaly label. Send this ONNX into
OpenExplorer for quantization and compilation.

## 03_export_calibration_images.py

Exports representative normal images for OpenExplorer PTQ calibration.

## 04_rebuild_stats_from_bpu_features.py

After you get the `.hbm` or `.bin` model, run all normal images through the
quantized model on RDK X5 or in the simulator, dump the BPU output features to
`.npy`, `.npz`, or `.csv`, then run this script.

This produces `bpu_anomaly_stats.npz`, which is the statistics file you should
use on RDK X5.

## 05_predict_float_check.py

PC-only sanity check with the PyTorch float model. It is not the final RDK X5
runtime path.

## 06_predict_float_visual.py

Run this after `01_train_float_stats.py`.

It uses `float_anomaly_stats.npz` to predict a test image folder on PC, saves a
CSV result file, and creates `pc_predict_summary.jpg` for quick visual checking.

## Runtime idea on RDK X5

1. Load `.hbm` or `.bin`.
2. Preprocess image exactly like the metadata says.
3. Run BPU inference.
4. Read the output feature tensor.
5. If the output is raw feature, do L2 normalization.
6. Load `bpu_anomaly_stats.npz`.
7. Compute z-score RMS and compare with threshold.

### 总结

通过runs 对比可以发现，Resnet18 比 Mobilenet_v2 具有更好的特征提取能力

### Notice!

裁剪的代码，AI通常会生成为

```python
def center_crop_resize_rgb(image_path: Path, input_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    image = image.crop((left, top, left + short_side, top + short_side))
    return image.resize((input_size, input_size), Image.Resampling.BILINEAR)


但这个是从中心裁剪成正方形，因此就会导致边缘被裁剪，因此当缺陷在边缘的时候，缺陷就无法被正常检测
