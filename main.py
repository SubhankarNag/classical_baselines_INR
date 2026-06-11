import os
import argparse
import numpy as np

from jpeg_codec import compress_jpeg, decompress_jpeg
from jpeg2000_codec import compress_jpeg2000, decompress_jpeg2000
from video_codec import compress_video, decompress_video
from utils import load_vol

OCMR_MAT_DATA_DIR = "../../dataset/OCMR_data/"

def get_folder_size(path, extensions=None):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                total += os.path.getsize(os.path.join(root, f))
    return total

def compression_ratio(original_file, compressed_folder, codec):
    original_size = os.path.getsize(original_file)
    
    ext_map = {"jpeg": [".jpg"], "jpeg2000": [".jp2"], "h264": [".mp4"], "h265": [".mp4"]}
    exts = ext_map.get(codec.lower())

    compressed_size = get_folder_size(compressed_folder, extensions=exts)

    ratio = original_size / compressed_size

    print("\n==========")
    print("Original Size   :", original_size)
    print("Compressed Size :", compressed_size)
    print("Compression Ratio:", ratio)
    print("==========\n")

    return ratio

def calculate_rrmse(original_file, recon_file):
    original = load_vol(original_file)
    reconstructed = np.load(recon_file)

    # RRMSE = ||orig - recon||_2 / ||orig||_2
    rrmse = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

    print("RRMSE           :", rrmse)
    print("==========\n")

def run_jpeg(input_file, output_dir, quality):
    print("Compressing JPEG...")

    compress_jpeg(npy_file=input_file, out_dir=output_dir, quality=quality)

    recon_file = os.path.join(output_dir, "reconstructed.npy")

    print("Reconstructing JPEG...")

    decompress_jpeg(compressed_dir=output_dir, output_npy=recon_file)

    compression_ratio(input_file, output_dir, "jpeg")

    calculate_rrmse(input_file, recon_file)

    print("Saved:")
    print(recon_file)


def run_jpeg2000(input_file, output_dir, cratio):
    print("Compressing JPEG2000...")

    compress_jpeg2000(npy_file=input_file, out_dir=output_dir, cratio=cratio)

    recon_file = os.path.join(output_dir, "reconstructed.npy")

    print("Reconstructing JPEG2000...")

    decompress_jpeg2000(compressed_dir=output_dir, output_npy=recon_file)

    compression_ratio(input_file, output_dir, "jpeg2000")

    calculate_rrmse(input_file, recon_file)

    print("Saved:")
    print(recon_file)


def run_video(input_file, output_dir, codec, crf):
    print(f"Compressing {codec}...")

    compress_video(npy_file=input_file, out_dir=output_dir, codec=codec, crf=crf)

    recon_file = os.path.join(output_dir, "reconstructed.npy")

    print(f"Reconstructing {codec}...")

    decompress_video(compressed_dir=output_dir, output_npy=recon_file)

    compression_ratio(input_file, output_dir, codec)

    calculate_rrmse(input_file, recon_file)

    print("Saved:")
    print(recon_file)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--codec", type=str, required=True, choices=["jpeg", "jpeg2000", "h264", "h265"]
    )

    parser.add_argument("--input", type=str, required=True)

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--quality", type=int, default=50)

    parser.add_argument("--cratio", type=int, default=20)

    parser.add_argument("--crf", type=int, default=23)

    args = parser.parse_args()

    args.input = os.path.join(OCMR_MAT_DATA_DIR, args.input)

    if args.codec == "jpeg":

        run_jpeg(args.input, args.output, args.quality)

    elif args.codec == "jpeg2000":

        run_jpeg2000(args.input, args.output, args.cratio)

    elif args.codec in ["h264", "h265"]:

        run_video(args.input, args.output, args.codec, args.crf)


if __name__ == "__main__":
    main()
