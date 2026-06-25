import os
import argparse
import csv
import torch
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# from jpeg_codec import compress_jpeg_from_vol, decompress_jpeg
# from jpeg2000_codec import compress_jpeg2000_from_vol, decompress_jpeg2000
from video_codec import compress_video_from_vol, decompress_video

# from spiht_codec import compress_spiht_from_vol, decompress_spiht


from utils import load_vol
from eval import compute_rrmse, compute_batched_metrics
from plotting import visualize_and_save
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Configuration derived from run_rd_curve.py and eval_rd.py
OCMR_MAT_DATA_DIR = "../../dataset/OCMR_data/"
# JPEG_QUALITIES = [1, 3, 5, 7, 10, 13, 16, 19] #[20, 35, 50, 65, 80] #[10, 20, 30, 40, 50, 60, 70, 80, 90]
# CRATIOS = [80, 110, 130, 160, 183, 205, 228, 250] #[20, 35, 50, 65, 80] #[5, 10, 20, 40, 80, 160]
# CRFS = [25, 26, 27, 28, 29, 30, 31, 32, 33, 35] #[23, 29, 36, 42, 48] #[18, 23, 28, 33, 38, 43, 48, 51]
# bitrate
CRFS = ["10K", "20K", "30K", "40K", "50K", "60K", "70K", "80K", "90K", "100K", "130K", "200k"]

CODECS = ["h266"] #["jpeg", "jpeg2000", "h264", "h265"]

# CODECS = ["spiht"]
# SPIHT_BPPS = [
#     # 16.0,
#     # 8.0,
#     # 4.0,
#     # 2.0,
#     # 1.0,
#     0.7,
#     0.6,
#     0.5,
#     0.4,
#     0.3,
#     0.2,
#     0.1,
#     0.08,
#     0.06,
#     0.04,
#     0.02,
# ]

def get_folder_size(path, extensions=None):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                total += os.path.getsize(os.path.join(root, f))
    return total


# def get_folder_size(path, extensions=None):
#     total = 0

#     for root, _, files in os.walk(path):
#         for f in files:
#             include_file = (
#                 extensions is None
#                 or any(f.lower().endswith(ext) for ext in extensions)
#                 or f.lower() == "meta.pkl"
#             )

#             if include_file:
#                 total += os.path.getsize(os.path.join(root, f))

#     return total

def derive_complex_and_rsos(vol_ri):
    """Derived complex volume and RSOS from RI representation (W, H, D, C, T, 2)."""
    vol_complex = vol_ri[..., 0] + 1j * vol_ri[..., 1]
    rsos = np.sqrt(np.sum(np.abs(vol_complex)**2, axis=3))
    return vol_complex.astype(np.complex64), rsos.astype(np.float32)

def plot_all_rd_curves(results_list, output_dir):
    """Generates RD curve plots. Aggregates metrics across multiple files if present."""
    metrics = ["RRMSE", "PSNR", "SSIM", "LPIPS"]
    levels = ["Vol", "RSOS"]
    time_metrics = ["Comp_Time", "Decomp_Time"]
    
    codecs_found = sorted(list(set(r["Codec"] for r in results_list)))
    
    # Aggregation logic: average metrics across all files for each (Codec, Param)
    aggregated_results = []
    for codec in codecs_found:
        codec_entries = [r for r in results_list if r["Codec"] == codec]
        params_found = sorted(list(set(e["Param"] for e in codec_entries)))
        for p in params_found:
            param_entries = [e for e in codec_entries if e["Param"] == p]
            avg_res = {"Codec": codec, "Param": p}
            keys_to_avg = ["Ratio", "Comp_Time", "Decomp_Time"] + \
                          [f"{l}_{m}" for l in levels for m in metrics]
            for k in keys_to_avg:
                avg_res[k] = np.mean([e[k] for e in param_entries])
            aggregated_results.append(avg_res)

    # Plot Quality Metrics
    for level in levels:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            col_name = f"{level}_{metric}"
            for codec in codecs_found:
                codec_data = [r for r in aggregated_results if r["Codec"] == codec]
                codec_data.sort(key=lambda x: x["Ratio"])
                plt.plot([r["Ratio"] for r in codec_data], [r[col_name] for r in codec_data], marker='o', label=codec.upper())
            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"RD Curve: {level} {metric} (Averaged across files)")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"rd_curve_{level}_{metric}.png"))
            plt.close()

    # Plot Timing Metrics
    for tm in time_metrics:
        plt.figure(figsize=(10, 6))
        for codec in codecs_found:
            codec_data = [r for r in aggregated_results if r["Codec"] == codec]
            codec_data.sort(key=lambda x: x["Ratio"])
            plt.plot([r["Ratio"] for r in codec_data], [r[tm] for r in codec_data], marker='o', label=codec.upper())
        plt.xlabel("Compression Ratio")
        plt.ylabel("Time (seconds)")
        plt.title(f"{tm} vs Compression Ratio (Averaged)")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"rd_curve_{tm}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive RD Curve Benchmarking V2")
    parser.add_argument("--files", nargs="+", required=True, help="Filenames in OCMR_data (e.g. fs_0045_3T.h5)")
    parser.add_argument("--output_dir", type=str, default="results_rd_v2_H266")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Perceptual Metric
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "rd_metrics_comprehensive.csv")
    
    csv_headers = [
        "Filename", "Codec", "Param", "Ratio",
        "Comp_Time", "Decomp_Time",
        "Vol_RRMSE", "Vol_PSNR", "Vol_SSIM", "Vol_LPIPS",
        "RSOS_RRMSE", "RSOS_PSNR", "RSOS_SSIM", "RSOS_LPIPS"
    ]

    all_results = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric columns for plotting and skip-logic
                    for k in ["Ratio", "Comp_Time", "Decomp_Time", "Vol_RRMSE", "Vol_PSNR", "Vol_SSIM", "Vol_LPIPS", "RSOS_RRMSE", "RSOS_PSNR", "RSOS_SSIM", "RSOS_LPIPS"]:
                        if k in row: row[k] = float(row[k])
                    try: row["Param"] = int(row["Param"])
                    except:
                        try: row["Param"] = float(row["Param"])
                        except: pass
                    all_results.append(row)
            print(f"Loaded {len(all_results)} existing results from {csv_path}")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()

        # for each file 
        for filename in args.files:
            input_path = os.path.join(OCMR_MAT_DATA_DIR, filename)
            if not os.path.exists(input_path):
                print(f"Skipping {filename}: Not found in {OCMR_MAT_DATA_DIR}")
                continue

            print(f"\n[FILE] Loading original: {filename}")
            orig_ri = load_vol(input_path)
            orig_complex, orig_rsos = derive_complex_and_rsos(orig_ri)
            orig_size = os.path.getsize(input_path)
            W, H, D, C, T = orig_complex.shape

            # for each type of compression 
            for codec in CODECS:
                params = JPEG_QUALITIES if codec == "jpeg" else \
                         CRATIOS if codec == "jpeg2000" else \
                         CRFS if codec in ["h264", "h265", "h266"] else SPIHT_BPPS

                for p in params:
                    tag = f"{codec}_{p}"
                    if any(r["Filename"] == filename and r["Codec"] == codec and r["Param"] == p for r in all_results):
                        print(f"  Skipping {filename} | {tag}: Already in results.")
                        continue

                    print(f"  Benchmarking {filename} | {tag}...")
                    
                    work_dir = Path(args.output_dir) / filename.replace(".h5", "") / tag
                    work_dir.mkdir(parents=True, exist_ok=True)
                    recon_npz = work_dir / "reconstructed.npz"

                    # 1. Pipeline: Compress then Decompress with Timing
                    try:
                        if codec == "jpeg":
                            s_time = time.time()
                            compress_jpeg_from_vol(orig_ri, work_dir, quality=p)
                            comp_time = time.time() - s_time
                            
                            s_time = time.time()
                            decompress_jpeg(work_dir, recon_npz)
                            decomp_time = time.time() - s_time
                            exts = [".jpg"]
                        elif codec == "jpeg2000":
                            s_time = time.time()
                            compress_jpeg2000_from_vol(orig_ri, work_dir, cratio=p)
                            comp_time = time.time() - s_time
                            
                            s_time = time.time()
                            decompress_jpeg2000(work_dir, recon_npz)
                            decomp_time = time.time() - s_time
                            exts = [".jp2"]
                        elif codec == "spiht":
                            s_time = time.time()
                            compress_spiht_from_vol(
                                orig_ri,
                                work_dir,
                                bpp=p
                            )
                            comp_time = time.time() - s_time
                            
                            s_time = time.time()
                            decompress_spiht(
                                work_dir,
                                recon_npz
                            )
                            decomp_time = time.time() - s_time
                            exts = [".spiht"]
                        else: # h264, h265
                            s_time = time.time()
                            compress_video_from_vol(orig_ri, work_dir, codec=codec, crf=p)
                            comp_time = time.time() - s_time
                            
                            s_time = time.time()
                            decompress_video(work_dir, recon_npz, codec=codec)
                            decomp_time = time.time() - s_time
                            if codec == "h266":
                                exts = [".mkv"]
                            else:
                                exts = [".mp4"]
                        
                    except Exception as e:
                        print(f"    Error processing {tag}: {e}")
                        continue

                    # 2. Compression Ratio
                    comp_size = get_folder_size(work_dir, extensions=exts)
                    ratio = orig_size / comp_size if comp_size > 0 else 0

                    # 3. Process Reconstruction
                    recon_ri = np.load(recon_npz)["data"].astype(np.float32)
                    recon_complex, recon_rsos = derive_complex_and_rsos(recon_ri)

                    # 4. Metric Computation
                    # Volume Level (Magnitude)
                    t_orig_vol = torch.from_numpy(np.abs(orig_complex)).permute(3, 4, 0, 1, 2).reshape(C*T, W, H, D)
                    t_recon_vol = torch.from_numpy(np.abs(recon_complex)).permute(3, 4, 0, 1, 2).reshape(C*T, W, H, D)
                    vol_rrmse = compute_rrmse(t_orig_vol, t_recon_vol)
                    vol_psnr, vol_ssim, vol_lpips = compute_batched_metrics(t_orig_vol, t_recon_vol, lpips_metric, device)

                    # RSOS Level
                    t_orig_rsos = torch.from_numpy(orig_rsos).permute(3, 0, 1, 2)
                    t_recon_rsos = torch.from_numpy(recon_rsos).permute(3, 0, 1, 2)
                    rsos_rrmse = compute_rrmse(t_orig_rsos, t_recon_rsos)
                    rsos_psnr, rsos_ssim, rsos_lpips = compute_batched_metrics(t_orig_rsos, t_recon_rsos, lpips_metric, device)

                    # 5. Visual Comparison (Central slice/time)
                    visualize_and_save(
                        orig_complex, recon_complex, orig_rsos, recon_rsos,
                        t_idx=T//2, d_idx=D//2, c_idx=0,
                        save_path=str(work_dir / f"plot_{tag}.png")
                    )

                    # 6. Record Results
                    res = {
                        "Filename": filename, "Codec": codec, "Param": p, "Ratio": ratio,
                        "Comp_Time": comp_time, "Decomp_Time": decomp_time,
                        "Vol_RRMSE": vol_rrmse, "Vol_PSNR": vol_psnr, "Vol_SSIM": vol_ssim, "Vol_LPIPS": vol_lpips,
                        "RSOS_RRMSE": rsos_rrmse, "RSOS_PSNR": rsos_psnr, "RSOS_SSIM": rsos_ssim, "RSOS_LPIPS": rsos_lpips
                    }
                    writer.writerow(res)
                    all_results.append(res)
                    f.flush()


    if all_results:
        plot_all_rd_curves(all_results, args.output_dir)
        print(f"\nBenchmarking complete. Results and plots saved to {args.output_dir}")
    else:
        print("No results were generated. Check file paths and codec installations.")

if __name__ == "__main__":
    main()