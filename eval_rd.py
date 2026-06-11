import os, argparse, csv, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from jpeg_codec import compress_jpeg, decompress_jpeg
from jpeg2000_codec import compress_jpeg2000, decompress_jpeg2000
from video_codec import compress_video, decompress_video
from utils import load_vol
from eval import compute_rrmse, compute_batched_metrics
from plotting import visualize_and_save
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Evaluation Configuration
OCMR_MAT_DATA_DIR = "../../dataset/OCMR_data/"
JPEG_QUALITIES = [10, 20]#, 30, 40, 50, 60, 70, 80, 90]
JPEG2000_CRATIOS = [18, 23]#, 28, 33, 38, 43, 48, 51]
VIDEO_CRFS = [5, 10]#, 20, 40, 80, 160]

CODECS = ["jpeg", "jpeg2000", "h264", "h265"]

def get_folder_size(path, extensions=None):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                total += os.path.getsize(os.path.join(root, f))
    return total

def derive_complex_and_rsos(vol_ri):
    """Derived complex volume and RSOS from RI representation (W, H, D, C, T, 2)."""
    vol_complex = vol_ri[..., 0] + 1j * vol_ri[..., 1]
    rsos = np.sqrt(np.sum(np.abs(vol_complex)**2, axis=3))
    return vol_complex.astype(np.complex64), rsos.astype(np.float32)

def plot_all_rd_curves(results_list, output_dir):
    """Generates RD curve plots for RRMSE, PSNR, SSIM, and LPIPS."""
    metrics = ["RRMSE", "PSNR", "SSIM", "LPIPS"]
    levels = ["Vol", "RSOS"]
    
    for level in levels:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            col_name = f"{level}_{metric}"
            
            # Group results by codec for plotting
            codecs_found = sorted(list(set(r["Codec"] for r in results_list)))
            for codec in codecs_found:
                codec_data = [r for r in results_list if r["Codec"] == codec]
                # Sort by ratio for a clean line plot
                codec_data.sort(key=lambda x: x["Ratio"])
                
                ratios = [r["Ratio"] for r in codec_data]
                vals = [r[col_name] for r in codec_data]
                plt.plot(ratios, vals, marker='o', label=codec.upper())

            plt.xlabel("Compression Ratio")
            plt.ylabel(metric)
            plt.title(f"RD Curve: {level} {metric}")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"rd_curve_{level}_{metric}.png"))
            plt.close()

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Names of .h5 files in OCMR_data")
    parser.add_argument("--output_dir", type=str, default="rd_eval_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "rd_metrics_comprehensive.csv")
    
    csv_headers = [
        "Filename", "Codec", "Param", "Ratio",
        "Vol_RRMSE", "Vol_PSNR", "Vol_SSIM", "Vol_LPIPS",
        "RSOS_RRMSE", "RSOS_PSNR", "RSOS_SSIM", "RSOS_LPIPS"
    ]

    all_metrics_data = []
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for filename in args.files:
            input_path = os.path.join(OCMR_MAT_DATA_DIR, filename)
            if not os.path.exists(input_path): continue

            # Load original once per file to avoid redundant I/O
            print(f"\n[FILE] Processing original volume: {filename}")
            orig_ri = load_vol(input_path) 
            orig_complex, orig_rsos = derive_complex_and_rsos(orig_ri)
            orig_size = os.path.getsize(input_path)
            # Masking is not needed
            W, H, D, C, T = orig_complex.shape
            
            for codec in CODECS:
                params = JPEG_QUALITIES if codec == "jpeg" else \
                         JPEG2000_CRATIOS if codec == "jpeg2000" else VIDEO_CRFS

                for p in params:
                    tag = f"{codec}_{p}"
                    print(f"  Evaluating {tag}...")
                    
                    work_dir = Path(args.output_dir) / filename.replace(".h5", "") / tag
                    work_dir.mkdir(parents=True, exist_ok=True)
                    recon_npy = work_dir / "reconstructed.npy"

                    # 1. Pipeline: Compress then Decompress
                    if codec == "jpeg":
                        compress_jpeg(input_path, work_dir, quality=p)
                        decompress_jpeg(work_dir, recon_npy)
                        exts = [".jpg"]
                    elif codec == "jpeg2000":
                        compress_jpeg2000(input_path, work_dir, cratio=p)
                        decompress_jpeg2000(work_dir, recon_npy)
                        exts = [".jp2"]
                    else:
                        compress_video(input_path, work_dir, codec=codec, crf=p)
                        decompress_video(work_dir, recon_npy)
                        exts = [".mp4"]

                    # 2. Compression Ratio
                    comp_size = get_folder_size(work_dir, extensions=exts)
                    ratio = orig_size / comp_size if comp_size > 0 else 0

                    # 3. Load & Process Reconstruction
                    recon_ri = np.load(recon_npy).astype(np.float32)
                    recon_complex, recon_rsos = derive_complex_and_rsos(recon_ri)

                    # 4. Metric Computation
                    # Volume Level
                    t_orig_vol = torch.from_numpy(np.abs(orig_complex)).permute(3, 4, 0, 1, 2).reshape(C*T, W, H, D)
                    t_recon_vol = torch.from_numpy(np.abs(recon_complex)).permute(3, 4, 0, 1, 2).reshape(C*T, W, H, D)
                    vol_rrmse = compute_rrmse(t_orig_vol, t_recon_vol)
                    vol_psnr, vol_ssim, vol_lpips = compute_batched_metrics(t_orig_vol, t_recon_vol, lpips_metric, device)

                    # RSOS Level
                    t_orig_rsos = torch.from_numpy(orig_rsos).permute(3, 0, 1, 2)
                    t_recon_rsos = torch.from_numpy(recon_rsos).permute(3, 0, 1, 2)
                    rsos_rrmse = compute_rrmse(t_orig_rsos, t_recon_rsos)
                    rsos_psnr, rsos_ssim, rsos_lpips = compute_batched_metrics(t_orig_rsos, t_recon_rsos, lpips_metric, device)

                    # 5. Save Comparison Plot (Central slice/time)
                    visualize_and_save(
                        orig_complex, recon_complex, orig_rsos, recon_rsos,
                        t_idx=T//2, d_idx=D//2, c_idx=0,
                        save_path=str(work_dir / f"plot_{tag}.png")
                    )

                    # 6. Record Results
                    res_row = {
                        "Filename": filename, "Codec": codec, "Param": p, "Ratio": ratio,
                        "Vol_RRMSE": vol_rrmse, "Vol_PSNR": vol_psnr, "Vol_SSIM": vol_ssim, "Vol_LPIPS": vol_lpips,
                        "RSOS_RRMSE": rsos_rrmse, "RSOS_PSNR": rsos_psnr, "RSOS_SSIM": rsos_ssim, "RSOS_LPIPS": rsos_lpips
                    }
                    writer.writerow(res_row)
                    all_metrics_data.append(res_row)
                    f.flush()

                    # 7. Cleanup
                    if os.path.exists(recon_npy): os.remove(recon_npy)

    plot_all_rd_curves(all_metrics_data, args.output_dir)

    print(f"\nEvaluation finished. Metrics saved to {csv_path}")

if __name__ == "__main__":
    run_evaluation()
