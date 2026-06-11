import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import math
from tqdm.auto import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# import sys
# PROJECT_ROOT = os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__))
# )
# sys.path.insert(0, PROJECT_ROOT)

from read_ocmr import read_ocmr


def kspace_to_image_2d(kspace):
    """2D IFFT over kx, ky with proper shifts."""
    image = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=[0, 1]),
            axes=[0, 1],
        ),
        axes=[0, 1],
    )
    return image


def load_original_ocmr_rsos(h5_path):
    """
    Load OCMR .h5 k-space, apply 2D IFFT, retain (kx, ky, slice, coil, phase),
    compute RSOS magnitude → (W, H, D, T) float32 tensor.
    """
    try:
        kData, param = read_ocmr(h5_path)
        print(f"  Raw kData shape: {kData.shape}")
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None

    # Average over avg dim (axis 8)
    kData_tmp = np.mean(kData, axis=8)  # → (kx, ky, kz, coil, phase, set, slice, rep)

    # 2D IFFT over kx, ky
    im_coil = kspace_to_image_2d(kData_tmp)

    # Retain only (kx, ky, slice, coil, phase) — remove kz(0), set(0), rep(0)
    im_coil = im_coil[:, :, 0, :, :, 0, :, 0]  # → (kx, ky, coil, phase, slice)
    im_coil = np.transpose(im_coil, (0, 1, 4, 2, 3))  # → (kx, ky, slice, coil, phase)

    # RSOS: sqrt(sum |coil|^2) over coil dim (axis 3) → (W, H, D, T)
    rsos = np.sqrt(np.sum(np.abs(im_coil) ** 2, axis=3)).astype(np.float32)

    return torch.tensor(im_coil), torch.tensor(rsos, dtype=torch.float32)


def load_recon_ocmr_rsos(path):
    data = np.load(path)
    image = data["image"]
    rsos = data["rsos"]

    return torch.from_numpy(image), torch.from_numpy(rsos)


# --- METRIC COMPUTATION ---


def compute_rrmse(original, comparison):
    original = original.cpu().to(torch.float64)
    comparison = comparison.cpu().to(torch.float64)
    error_rmse = torch.sqrt(torch.mean((original - comparison) ** 2))
    signal_rmse = torch.sqrt(torch.mean(original**2))
    if signal_rmse < 1e-10:
        return 0.0
    return (error_rmse / signal_rmse).item()


def compute_batched_metrics(tensor1, tensor2, lpips_metric, device, batch_size=32):
    """Computes PSNR, SSIM, LPIPS in batches."""
    if tensor1.ndim == 3:
        t1 = tensor1.permute(2, 0, 1).unsqueeze(1)
        t2 = tensor2.permute(2, 0, 1).unsqueeze(1)
    elif tensor1.ndim == 4:
        T, X, Y, Z = tensor1.shape
        t1 = tensor1.permute(0, 3, 1, 2).reshape(-1, 1, X, Y)
        t2 = tensor2.permute(0, 3, 1, 2).reshape(-1, 1, X, Y)

    N = t1.shape[0]
    data_range = tensor1.max() - tensor1.min()
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    min_val, max_val = tensor1.min(), tensor1.max()

    with torch.no_grad():
        for i in range(0, N, batch_size):
            b1 = t1[i : i + batch_size].to(device)
            b2 = t2[i : i + batch_size].to(device)
            current_batch_size = b1.shape[0]

            total_psnr += psnr_metric(b2, b1).item() * current_batch_size
            total_ssim += ssim_metric(b2, b1).item() * current_batch_size

            b1_norm = (b1 - min_val) / (max_val - min_val)
            b2_norm = (b2 - min_val) / (max_val - min_val)

            b1_lpips = torch.clamp((b1_norm * 2.0) - 1.0, -1.0, 1.0)
            b2_lpips = torch.clamp((b2_norm * 2.0) - 1.0, -1.0, 1.0)

            b1_rgb = b1_lpips.repeat(1, 3, 1, 1)
            b2_rgb = b2_lpips.repeat(1, 3, 1, 1)

            total_lpips += lpips_metric(b1_rgb, b2_rgb).sum().item()
            count += current_batch_size

    return total_psnr / count, total_ssim / count, total_lpips / count


def compute_per_slice_metrics(vol_orig, vol_rec, lpips_metric, device, batch_size=32):
    """
    Computes per-slice (over all phases) metrics.
    vol_orig, vol_rec: (W, H, D, T).
    Returns: list of dicts, one per slice.
    """
    W, H, D, T = vol_orig.shape
    per_slice = []

    for s in range(D):
        # Single slice: (W, H, T)
        s_orig = vol_orig[:, :, s, :]  # (W, H, T)
        s_rec = vol_rec[:, :, s, :]

        rrmse = compute_rrmse(s_orig, s_rec)

        # Reshape to (T, 1, W, H) for batched metrics
        t1 = s_orig.permute(2, 0, 1).unsqueeze(1)  # (T, 1, W, H)
        t2 = s_rec.permute(2, 0, 1).unsqueeze(1)

        data_range = s_orig.max() - s_orig.min()
        if data_range < 1e-10:
            data_range = torch.tensor(1.0)

        psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

        min_val, max_val = s_orig.min(), s_orig.max()
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        count = 0

        with torch.no_grad():
            for i in range(0, T, batch_size):
                b1 = t1[i : i + batch_size].to(device)
                b2 = t2[i : i + batch_size].to(device)
                bs = b1.shape[0]

                total_psnr += psnr_metric(b2, b1).item() * bs
                total_ssim += ssim_metric(b2, b1).item() * bs

                b1_n = (b1 - min_val) / (max_val - min_val + 1e-10)
                b2_n = (b2 - min_val) / (max_val - min_val + 1e-10)
                b1_l = torch.clamp(b1_n * 2.0 - 1.0, -1.0, 1.0).repeat(1, 3, 1, 1)
                b2_l = torch.clamp(b2_n * 2.0 - 1.0, -1.0, 1.0).repeat(1, 3, 1, 1)
                total_lpips += lpips_metric(b1_l, b2_l).sum().item()
                count += bs

        if count == 0:
            per_slice.append(
                {"Slice": s, "PSNR": 0.0, "SSIM": 0.0, "RRMSE": 0.0, "LPIPS": 0.0}
            )
        else:
            per_slice.append(
                {
                    "Slice": s,
                    "PSNR": total_psnr / count,
                    "SSIM": total_ssim / count,
                    "RRMSE": rrmse,
                    "LPIPS": total_lpips / count,
                }
            )

    return per_slice


def execute(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing LPIPS Metric...")
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=False
    ).to(device)

    orig_files = [
        "fs_0045_3T.h5",
        "fs_0095_1_5T.h5",
        'fs_0069_1_5T.h5',
        'fs_0057_1_5T.h5',
        'fs_0056_1_5T.h5',
        'fs_0060_1_5T.h5',
        'fs_0068_1_5T.h5',
        'fs_0074_1_5T.h5',
        'fs_0053_1_5T.h5',
        'fs_0063_1_5T.h5',
        'fs_0012_3T.h5',
    ]

    rec_files = []
    for f in orig_files:
        tmp = f"{f.split('.')[0]}.nii.npz"
        rec_files.append(tmp)

    vol_results = []
    rsos_results = []

    print(f"Starting Evaluation on {len(orig_files)} files...")

    for orig_f, rec_f in tqdm(zip(orig_files, rec_files), total=len(orig_files)):
        orig_path = os.path.join(args.original_dir, orig_f)
        rec_path = os.path.join(args.reconstructed_dir, rec_f)

        # Load Data (Robust Loader)
        vol_orig, vol_orig_rsos = load_original_ocmr_rsos(
            orig_path
        )  # (kx, ky, slice, coil, phase), (W,H,D,T)

        vol_rec, vol_rec_rsos = load_recon_ocmr_rsos(rec_path)  # (W,H,D,C,T), (W,H,D,T)

        W, H, D, C, T = vol_orig.shape

        if vol_orig.shape != vol_rec.shape:
            print(
                f"  Shape mismatch: orig={vol_orig.shape}, rec={vol_rec.shape}. Skipping."
            )
            continue

        vol_orig = (
            torch.abs(vol_orig).permute(3, 4, 0, 1, 2).reshape(C * T, W, H, D)
        )  # (C,T,W,H,D) -> (C×T, W, H, D)
        vol_rec = torch.abs(vol_rec).permute(3, 4, 0, 1, 2).reshape(C * T, W, H, D)

        if vol_orig is None or vol_rec is None:
            continue

        # --- 1. VOLUME METRICS ---
        rrmse = compute_rrmse(vol_orig, vol_rec)
        psnr, ssim, lpips = compute_batched_metrics(
            vol_orig, vol_rec, lpips_metric, device, batch_size=32
        )

        vol_results.append(
            {"File": rec_f, "PSNR": psnr, "SSIM": ssim, "RRMSE": rrmse, "LPIPS": lpips}
        )

        # --- 2. CORRELATION MAP METRICS ---
        vol_orig_rsos = vol_orig_rsos.permute(3, 0, 1, 2)  # (W,H,D,T) -> (T,W,H,D)
        vol_rec_rsos = vol_rec_rsos.permute(3, 0, 1, 2)  # (W,H,D,T) -> (T,W,H,D)

        rrmse = compute_rrmse(vol_orig_rsos, vol_rec_rsos)
        psnr, ssim, lpips = compute_batched_metrics(
            vol_orig_rsos, vol_rec_rsos, lpips_metric, device, batch_size=32
        )

        rsos_results.append(
            {"File": rec_f, "PSNR": psnr, "SSIM": ssim, "RRMSE": rrmse, "LPIPS": lpips}
        )

        del vol_orig, vol_rec
        torch.cuda.empty_cache()

    os.makedirs(f"{args.reconstructed_dir}/csv_results", exist_ok=True)
    # 1. Volume Metrics
    if vol_results:
        df_vol_full = pd.DataFrame(vol_results)

        # Calculate Summary: Replace inf with NaN so mean calculation is valid
        df_clean = df_vol_full.replace([np.inf, -np.inf], np.nan)
        df_vol_mean = df_clean.mean(numeric_only=True)
        df_vol_std = df_clean.std(numeric_only=True)
        df_vol_summary = df_vol_mean.to_frame().T
        df_vol_summary.insert(0, "File", "Average")

        df_vol_meanstd = (
            (
                df_vol_mean.map(lambda m: f"{m:.4f}")
                + " ± "
                + df_vol_std.map(lambda s: f"{s:.4f}")
            )
            .to_frame()
            .T
        )
        df_vol_meanstd.insert(0, "File", "Average ± Std")

        print("\n--- Volume Metrics Summary (excluding inf) ---")
        print(df_vol_meanstd.to_string(index=False))

        full_path_vol = f"{args.reconstructed_dir}/csv_results/volume_metrics_table_full{args.suffix}.csv"
        summary_path_vol = f"{args.reconstructed_dir}/csv_results/volume_metrics_table{args.suffix}.csv"

        df_vol_full.to_csv(full_path_vol, index=False)
        df_vol_summary.to_csv(summary_path_vol, index=False)
        print(f"Saved Volume Metrics to:\n  {full_path_vol}\n  {summary_path_vol}")

    # 2. Map Metrics (With Individual Networks)
    if rsos_results:
        df_vol_full = pd.DataFrame(rsos_results)

        # Calculate Summary: Replace inf with NaN so mean calculation is valid
        df_clean = df_vol_full.replace([np.inf, -np.inf], np.nan)
        df_vol_mean = df_clean.mean(numeric_only=True)
        df_vol_std = df_clean.std(numeric_only=True)
        df_vol_summary = df_vol_mean.to_frame().T
        df_vol_summary.insert(0, "File", "Average")

        df_vol_meanstd = (
            (
                df_vol_mean.map(lambda m: f"{m:.4f}")
                + " ± "
                + df_vol_std.map(lambda s: f"{s:.4f}")
            )
            .to_frame()
            .T
        )
        df_vol_meanstd.insert(0, "File", "Average ± Std")

        print("\n--- RSOS Metrics Summary (excluding inf) ---")
        print(df_vol_meanstd.to_string(index=False))

        full_path_vol = f"{args.reconstructed_dir}/csv_results/rsos_metrics_table_full{args.suffix}.csv"
        summary_path_vol = f"{args.reconstructed_dir}/csv_results/rsos_metrics_table{args.suffix}.csv"

        df_vol_full.to_csv(full_path_vol, index=False)
        df_vol_summary.to_csv(summary_path_vol, index=False)
        print(f"Saved Volume Metrics to:\n  {full_path_vol}\n  {summary_path_vol}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Folder with original .mat files",
    )
    parser.add_argument(
        "--reconstructed_dir",
        type=str,
        required=True,
        help="Folder with reconstructed .mat files",
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder under outputs",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for output CSVs"
    )

    args = parser.parse_args()
    execute(args)


# python eval.py --original_dir '../../../dataset/OCMR_data/' --reconstructed_dir 'outputs/ultra_compression/' --folder 'ultra_compression'
