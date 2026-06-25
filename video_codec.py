import os, shutil, pickle, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_vol


FFMPEG = "/home/subhankar/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
# ffmpeg - this doesnt support h266

def compress_video(npy_file, out_dir, codec="h265", crf=23):
    kspace = load_vol(npy_file)
    compress_video_from_vol(kspace, out_dir, codec, crf)

def compress_video_from_vol(kspace, out_dir, codec="h265", crf=23):
    X,Y,Z,C,T,_ = kspace.shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    # ffcodec = "libx265" if codec.lower()=="h265" else "libx264"
    codec = codec.lower()

    if codec == "h264":
        ffcodec = "libx264"
    elif codec == "h265":
        ffcodec = "libx265"
    elif codec == "h266":
        ffcodec = "libvvenc"
    else:
        raise ValueError(codec)


    for z in tqdm(range(Z), desc=f"Compressing {codec}"):
        for c in range(C):
            for part,name in [(0,"r"),(1,"i")]:
                video = kspace[:,:,z,c,:,part]

                mn = float(video.min())
                mx = float(video.max())

                key = f"z{z:03d}_c{c:03d}_{name}"
                meta[key] = (mn,mx)

                tmp = out_dir/f"tmp_{key}"
                tmp.mkdir(exist_ok=True)

                for t in range(T):
                    frame = video[:,:,t]
                    frame = (frame-mn)/(mx-mn+1e-12)
                    frame = np.round(frame*65535).astype(np.uint16)

                    Image.fromarray(frame).save(tmp/f"frame_{t:04d}.png")

                # out_video = out_dir/f"{key}.mp4"
                ext = ".mkv" if codec == "h266" else ".mp4"
                out_video = out_dir / f"{key}{ext}"
                # Set pixel format dynamically based on codec compatibility
                pix_fmt_param = "yuv420p10le" if codec == "h266" else "gray16le"

                cmd = [
                    FFMPEG,
                    "-loglevel", "error",  # only errors
                    "-y",
                    "-framerate","10",
                    "-i",str(tmp/"frame_%04d.png"),
                    "-c:v",ffcodec,
                    # "-crf",str(crf),
                    "-b:v",str(crf),
                    "-bufsize", "1M",               # Allow rate-controller breathing room at low bitrates
                    "-fps_mode", "passthrough",     # Force every single input frame into the encoder
                    "-pix_fmt", pix_fmt_param,      # Others: gray16le; h266 - yuv420p10le
                    str(out_video)
                ]
                subprocess.run(cmd,check=True)

                shutil.rmtree(tmp)

    with open(out_dir/"meta.pkl","wb") as f:
        pickle.dump({"shape":kspace.shape,"meta":meta},f)

def decompress_video(compressed_dir, output_path, codec='h265'):
    compressed_dir = Path(compressed_dir)

    with open(compressed_dir/"meta.pkl","rb") as f:
        info = pickle.load(f)

    X,Y,Z,C,T,_ = info["shape"]
    meta = info["meta"]

    out = np.zeros((X,Y,Z,C,T,2), dtype=np.float32)

    for key, (mn, mx) in tqdm(meta.items(), desc="Decompressing video"):
        tmp = compressed_dir/f"decode_{key}"
        tmp.mkdir(exist_ok=True)

        # video_file = compressed_dir/f"{key}.mp4"
        video_file = (
            compressed_dir/f"{key}.mkv"
            if (compressed_dir/f"{key}.mkv").exists()
            else compressed_dir/f"{key}.mp4"
        )

        # pix_fmt_param = "yuv420p10le" if codec == "h266" else "gray16le"

        cmd = [
            FFMPEG,
            "-loglevel", "error",  # only errors
            "-y",
            "-i",str(video_file),
            "-fps_mode", "passthrough",     # Force extraction of every encoded frame
            "-pix_fmt", "gray16le",
            str(tmp/"frame_%04d.png")
        ]
        subprocess.run(cmd,check=True)

        toks = key.split("_")
        z = int(toks[0][1:])
        c = int(toks[1][1:])
        p = 0 if toks[2]=="r" else 1

        for t in range(T):
            frame = np.array(Image.open(tmp/f"frame_{t+1:04d}.png")).astype(np.float32)

            frame = frame/65535.0
            frame = frame*(mx-mn)+mn

            out[:,:,z,c,t,p] = frame

        shutil.rmtree(tmp)

    np.savez_compressed(output_path, data=out)
