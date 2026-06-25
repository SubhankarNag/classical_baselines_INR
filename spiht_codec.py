import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

from spiht import (
    SpihtSettings,
    encode_image,
    decode_image,
    EncodingResult,
)

from utils import load_vol

wavelet = "bior2.2"
quantization_scale = 1000.0
mode = "reflect"
level = None
color_space = None
per_channel_quant_scales = None

spiht_settings = SpihtSettings(
    wavelet,
    quantization_scale,
    mode,
    color_space,
    per_channel_quant_scales,
)


def compress_spiht(npy_file, out_dir, bpp=0.1):
    kspace = load_vol(npy_file)
    compress_spiht_from_vol(kspace, out_dir, bpp)


def compress_spiht_from_vol(kspace, out_dir, bpp=0.1):

    X, Y, Z, C, T, _ = kspace.shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    for z in tqdm(range(Z), desc="Compressing SPIHT"):
        for c in range(C):
            for t in range(T):
                for part, name in [(0, "r"), (1, "i")]:

                    img = kspace[:, :, z, c, t, part]

                    mn = float(img.min())
                    mx = float(img.max())

                    img_norm = (img - mn) / (mx - mn + 1e-12)
                    # img_u8 = np.round(img_norm * 255).astype(np.uint8)
                    img_u8 = img_norm.astype(np.float32)

                    # SPIHT expects (C,H,W)
                    img_u8 = img_u8[None, :, :]

                    h, w = img_u8.shape[1:]

                    if h % 2:
                        h -= 1
                    if w % 2:
                        w -= 1

                    img_u8 = img_u8[:, :h, :w]

                    max_bits = int(h * w * bpp)

                    encoded = encode_image(
                        img_u8,
                        spiht_settings,
                        level,
                        max_bits,
                    )

                    fname = f"z{z:03d}_c{c:03d}_t{t:03d}_{name}.spiht"

                    with open(out_dir / fname, "wb") as f:
                        f.write(encoded.encoded_bytes)

                    # Remove encoded_bytes from meta to avoid storing data twice
                    encoded_meta = encoded.to_dict()
                    encoded_meta.pop("encoding_result_encoded_bytes", None)

                    meta[fname] = {
                        "min": mn,
                        "max": mx,
                        "orig_shape": img.shape,
                        **encoded_meta
                    }

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(
            {
                "shape": kspace.shape,
                "meta": meta,
                "bpp": bpp,
            },
            f,
        )


def decompress_spiht(compressed_dir, output_path):

    compressed_dir = Path(compressed_dir)

    with open(compressed_dir / "meta.pkl", "rb") as f:
        info = pickle.load(f)

    X, Y, Z, C, T, _ = info["shape"]
    meta = info["meta"]

    out = np.zeros((X, Y, Z, C, T, 2), dtype=np.float32)

    for fname, entry in tqdm(
        meta.items(),
        desc="Decompressing SPIHT",
    ):

        with open(compressed_dir / fname, "rb") as f:
            encoded_bytes = f.read()

        # encoded = EncodingResult(encoded_bytes=encoded_bytes)
        enc_dict = {
            k: v
            for k, v in entry.items()
            if k.startswith("encoding_result_")
        }
        enc_dict["encoding_result_encoded_bytes"] = b"" # Placeholder for from_dict
        encoding_result = EncodingResult.from_dict(enc_dict)
        encoding_result.encoded_bytes = encoded_bytes
        

        decoded = decode_image(
            encoding_result,
            spiht_settings,
        )
        decoded = decoded[0].astype(np.float32)
        
        orig_x, orig_y = entry["orig_shape"]
        tmp = np.zeros((orig_x, orig_y), dtype=np.float32)
        tmp[:decoded.shape[0], :decoded.shape[1]] = decoded
        decoded = tmp
        
        

        mn = entry["min"]
        mx = entry["max"]

        # decoded = decoded / 255.0
        decoded = decoded * (mx - mn) + mn

        parts = fname.replace(".spiht", "").split("_")

        z = int(parts[0][1:])
        c = int(parts[1][1:])
        t = int(parts[2][1:])

        p = 0 if parts[3] == "r" else 1

        out[:, :, z, c, t, p] = decoded

    np.savez_compressed(output_path, data=out)
