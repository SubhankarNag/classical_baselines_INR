import pickle
from pathlib import Path

import numpy as np
import glymur
from tqdm import tqdm

from utils import load_vol

def compress_jpeg2000(npy_file, out_dir, cratio=20):
    kspace = load_vol(npy_file)
    compress_jpeg2000_from_vol(kspace, out_dir, cratio)

def compress_jpeg2000_from_vol(kspace, out_dir, cratio=20):
    X,Y,Z,C,T,_ = kspace.shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    for z in tqdm(range(Z), desc="Compressing JPEG2000"):
        for c in range(C):
            for t in range(T):
                for part,name in [(0,"r"),(1,"i")]:
                    img = kspace[:,:,z,c,t,part]

                    mn = float(img.min())
                    mx = float(img.max())

                    img_norm = (img - mn) / (mx - mn + 1e-12)
                    img_u16 = np.round(img_norm * 65535).astype(np.uint16)

                    fname = f"z{z:03d}_c{c:03d}_t{t:03d}_{name}.jp2"

                    glymur.Jp2k(
                        str(out_dir / fname),
                        data=img_u16,
                        cratios=[cratio]
                    )

                    meta[fname] = (mn,mx)

    with open(out_dir/"meta.pkl","wb") as f:
        pickle.dump(
            {
                "shape": kspace.shape,
                "meta": meta,
                "cratio": cratio
            },
            f
        )

def decompress_jpeg2000(compressed_dir, output_npy):
    compressed_dir = Path(compressed_dir)

    with open(compressed_dir/"meta.pkl","rb") as f:
        info = pickle.load(f)

    X,Y,Z,C,T,_ = info["shape"]
    meta = info["meta"]

    out = np.zeros((X,Y,Z,C,T,2), dtype=np.float32)

    for fname, (mn, mx) in tqdm(meta.items(), desc="Decompressing JPEG2000"):
        img_u16 = glymur.Jp2k(str(compressed_dir / fname))[:]

        img_norm = img_u16.astype(np.float32) / 65535.0
        img = img_norm * (mx - mn) + mn

        parts = fname.replace(".jp2","").split("_")

        z = int(parts[0][1:])
        c = int(parts[1][1:])
        t = int(parts[2][1:])
        p = 0 if parts[3] == "r" else 1

        out[:,:,z,c,t,p] = img

    np.save(output_npy, out)
