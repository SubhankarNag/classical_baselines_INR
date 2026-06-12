import os, json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_vol

def compress_jpeg(npy_file, out_dir, quality=50):
    kspace = load_vol(npy_file)
    compress_jpeg_from_vol(kspace, out_dir, quality)

def compress_jpeg_from_vol(kspace, out_dir, quality=50):
    X,Y,Z,C,T,_ = kspace.shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    for z in tqdm(range(Z), desc="Compressing JPEG"):
        for c in range(C):
            for t in range(T):
                for part,name in [(0,"r"),(1,"i")]:
                    img = kspace[:,:,z,c,t,part]

                    mn = float(img.min())
                    mx = float(img.max())

                    img_norm = (img - mn) / (mx - mn + 1e-12)
                    img_u8 = np.round(img_norm * 255).astype(np.uint8)

                    fname = f"z{z:03d}_c{c:03d}_t{t:03d}_{name}.jpg"
                    Image.fromarray(img_u8).save(out_dir / fname, quality=quality)

                    meta[fname] = (mn,mx)

    with open(out_dir/"meta.pkl","wb") as f:
        pickle.dump({"shape":kspace.shape,"meta":meta},f)

def decompress_jpeg(compressed_dir, output_path):
    compressed_dir = Path(compressed_dir)

    with open(compressed_dir/"meta.pkl","rb") as f:
        info = pickle.load(f)

    X,Y,Z,C,T,_ = info["shape"]
    meta = info["meta"]

    out = np.zeros((X,Y,Z,C,T,2), dtype=np.float32)

    for fname, (mn, mx) in tqdm(meta.items(), desc="Decompressing JPEG"):
        arr = np.array(Image.open(compressed_dir/fname)).astype(np.float32)

        arr = arr/255.0
        arr = arr*(mx-mn)+mn

        parts = fname.replace(".jpg","").split("_")
        z = int(parts[0][1:])
        c = int(parts[1][1:])
        t = int(parts[2][1:])
        p = 0 if parts[3]=="r" else 1

        out[:,:,z,c,t,p] = arr

    np.savez_compressed(output_path, data=out)
