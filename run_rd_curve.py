import os, argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from jpeg_codec import compress_jpeg, decompress_jpeg
from jpeg2000_codec import compress_jpeg2000, decompress_jpeg2000
from video_codec import compress_video, decompress_video
from utils import load_vol

OCMR_MAT_DATA_DIR = "../../dataset/OCMR_data/"
JPEG_QUALITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
CRFS = [18, 23, 28, 33, 38, 43, 48, 51]
CRATIOS = [5, 10, 20, 40, 80, 160]

def folder_size(path, extensions=None):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if extensions is None or any(f.lower().endswith(e) for e in extensions):
                total += os.path.getsize(os.path.join(root, f))
    return total

def calculate_rrmse(original, reconstructed):
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

def run_jpeg_rd(input_path, original_vol, out_root):
    results = []
    for q in tqdm(JPEG_QUALITIES, desc="JPEG RD Curve"):
        comp_dir = os.path.join(out_root, f"jpeg_q{q}")
        recon_path = os.path.join(comp_dir, "reconstructed.npz")
        compress_jpeg(input_path, comp_dir, q)
        decompress_jpeg(comp_dir, recon_path)
        
        recon = np.load(recon_path)["data"]
        ratio = os.path.getsize(input_path) / folder_size(comp_dir, [".jpg"])
        results.append({"param": q, "ratio": ratio, "rrmse": calculate_rrmse(original_vol, recon)})
    return results

def run_jpeg2000_rd(input_path, original_vol, out_root):
    results = []
    for cr in tqdm(CRATIOS, desc="JPEG2000 RD Curve"):
        comp_dir = os.path.join(out_root, f"j2k_cr{cr}")
        recon_path = os.path.join(comp_dir, "reconstructed.npz")
        compress_jpeg2000(input_path, comp_dir, cr)
        decompress_jpeg2000(comp_dir, recon_path)

        recon = np.load(recon_path)["data"]
        ratio = os.path.getsize(input_path) / folder_size(comp_dir, [".jp2"])
        results.append({"param": cr, "ratio": ratio, "rrmse": calculate_rrmse(original_vol, recon)})
    return results

def run_video_rd(input_path, original_vol, out_root, codec="h265"):
    results = []
    for crf in tqdm(CRFS, desc=f"{codec.upper()} RD Curve"):
        comp_dir = os.path.join(out_root, f"{codec}_crf{crf}")
        recon_path = os.path.join(comp_dir, "reconstructed.npz")
        compress_video(input_path, comp_dir, codec, crf)
        decompress_video(comp_dir, recon_path)

        recon = np.load(recon_path)["data"]
        ratio = os.path.getsize(input_path) / folder_size(comp_dir, [".mp4"])
        results.append({"param": crf, "ratio": ratio, "rrmse": calculate_rrmse(original_vol, recon)})
    return results

def plot_rd(all_data, out_path):
    plt.figure(figsize=(10, 6))
    for label, data in all_data.items():
        ratios = np.array([d['ratio'] for d in data])
        rrmses = np.array([d['rrmse'] for d in data])
        idx = np.argsort(ratios)
        plt.plot(ratios[idx], rrmses[idx], marker='o', label=label)
    
    plt.xlabel("Compression Ratio (Higher is better)")
    plt.ylabel("RRMSE (Lower is better)")
    plt.title("Rate-Distortion Performance Comparison")
    plt.legend(); plt.grid(True)
    plt.savefig(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Filename in OCMR_data")
    parser.add_argument("--output_dir", type=str, default="results_rd")
    args = parser.parse_args()

    input_full_path = os.path.join(OCMR_MAT_DATA_DIR, args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading original volume: {args.input}")
    original_vol = load_vol(input_full_path)
    
    all_results = {
        "JPEG": run_jpeg_rd(input_full_path, original_vol, args.output_dir),
        "JPEG2000": run_jpeg2000_rd(input_full_path, original_vol, args.output_dir)
    }

    # Add Video results if ffmpeg is available
    all_results.update({
        "H264": run_video_rd(input_full_path, original_vol, args.output_dir, "h264"),
        "H265": run_video_rd(input_full_path, original_vol, args.output_dir, "h265")
    })
    
    plot_rd(all_results, os.path.join(args.output_dir, "rd_curve.png"))
    
    csv_path = os.path.join(args.output_dir, "rd_metrics.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Codec", "Param", "Compression_Ratio", "RRMSE"])
        for codec, points in all_results.items():
            for p in points:
                writer.writerow([codec, p['param'], p['ratio'], p['rrmse']])
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
