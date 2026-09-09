
## Goal

The goal of this project is to reproduce and extend the method for dense metric depth completion from sparse direct Time-of-Flight (ToF) sensor measurements. Given an RGB image and a sparse depth map from a ToF sensor, the model predicts a dense, metric depth map.

## Features

- **Dual-Branch Encoder**: Separate encoders for RGB and depth modalities, fused via a MaskedJointAttention module that allows depth features to guide RGB feature learning while preventing RGB features from corrupting depth features.
- **RoPE Integration**: Utilizes Rotary Position Embedding (RoPE) compatible with DINOv3 backbones, applied per-layer inside the attention mechanism.
- **DPT Decoder**: Employs a DPT (Dense Prediction Transformer) decoder for high-resolution depth prediction.
- **Preprocessing Pipeline**: Includes scripts to load and preprocess the Hypersim dataset, converting Euclidean distance to z-depth and generating stage-specific downsampled and center-cropped RGB and ground-truth depth maps.
- **Training Script**: 
  - Supports single and multi-GPU training via DataParallel.
  - Comprehensive logging (CSV and TensorBoard) of training and validation metrics.
  - Checkpointing with automatic resumption and best-model saving based on validation delta1 accuracy.
  - Overfitting mode for debugging on a single sample.
  - Gradient clipping and learning rate scheduling with warmup and step decay.
- **Testing Script**: 
  - Runs inference on test data and saves predictions as .npy files.
  - Generates visualizations (ground truth, prediction, error maps) when matplotlib is available.
- **Modular Design**: 
  - Encoder (`encoder.py`), fusion module (`fusionv3.py`), decoder (`decoder.py`), loss computation (`losses.py`), and dataset handling (`preprocessed_dataset.py`, `hypersim_dataset.py`).
  - Normalization and intrinsic parameter utilities (`normalize.py`, `intrinsics.py`).
  - Alignment and augmentation utilities (`alignment.py`, `augment.py`).

## Usage

### Preprocessing
Run the preprocessing script to prepare the Hypersim dataset:
```bash
python main/preprocess.py
```

### Training
Start training with default settings:
```bash
python main/train.py
```
Key training arguments:
- `--batch-size`, `--enc_lr`, `--dec_lr`: Adjust batch size and learning rates.
- `--epochs`: Enable epoch-based training (set >0).
- `--use-tensorboard`: Enable TensorBoard logging.
- `--overfit`: Run overfitting on a single sample for debugging.
- `--resume`: Path to a checkpoint to resume training.

### Testing
Run inference on test data:
```bash
python main/train.py --test --checkpoint path/to/checkpoint.pt
```
Additional testing arguments:
- `--test-dir`: Directory containing test data (default: `hypersim_data_simple_nanfix`).
- `--test-save-dir`: Directory to save predictions (default: `test_predictions`).
- `--test-save-n`: Number of test predictions to save as .npy files (default: 3).

## Outputs
- Training logs: `logs/metrics.csv` (CSV) or `outputs/tb_logs/` (TensorBoard).
- Checkpoints: Saved in `outputs/checkpoints/` (including periodic, best, and final checkpoints).
- Predictions: During validation/testing, saved in `outputs/predictions/`.
- Visualizations: Generated during testing if matplotlib is installed.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- OpenCV
- Pillow
- matplotlib (for visualizations)
- tensorboard (optional, for TensorBoard logging)

# Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors

This project implements my interpretation of the approach described in the paper:

> **Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors**  
> Hakyeong Kim, Ruicheng Wang, Chengtang Yao, Jiaolong Yang, Min H. Kim  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 36518-36528  
> @InProceedings{Kim_2026_CVPR,
>    author = {Kim, Hakyeong and Wang, Ruicheng and Yao, Chengtang and Yang, Jiaolong and Kim, Min H.},
>    title = {Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors},
>    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
>    month = {June},
>    year = {2026},
>    pages = {36518-36528}
> }
