import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


# import sys
# PROJECT_ROOT = os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__))
# )
# sys.path.insert(0, PROJECT_ROOT)

from read_ocmr import read_ocmr


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def kspace_to_image_2d(kspace):
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=[0, 1]),
            axes=[0, 1],
        ),
        axes=[0, 1],
    )


def compute_rrmse(a, b, mask=None):
    """Compute RRMSE, optionally restricted to non-black (mask==True) pixels."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
    rmse = np.sqrt(np.mean((a - b) ** 2))
    denom = np.sqrt(np.mean(a ** 2)) + 1e-12
    return rmse / denom


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_original_ocmr(h5_path):
    kData, _ = read_ocmr(h5_path)
    kData_tmp = np.mean(kData, axis=8)
    im_coil = kspace_to_image_2d(kData_tmp)

    im_coil = im_coil[:, :, 0, :, :, 0, :, 0]
    im_coil = np.transpose(im_coil, (0, 1, 4, 2, 3))
    rsos = np.sqrt(np.sum(np.abs(im_coil) ** 2, axis=3))

    return im_coil, rsos  # (W,H,D,C,T), (W,H,D,T)


def load_complex_volume(path):
    with h5py.File(path, "r") as f:
        real = f["real"][()]
        imag = f["imag"][()]

    real = np.transpose(real, (4, 3, 2, 1, 0))
    imag = np.transpose(imag, (4, 3, 2, 1, 0))

    return real + 1j * imag  # (W,H,D,C,T)


def load_variance_volume(path):
    with h5py.File(path, "r") as f:
        real_var = f["real"][()]
        imag_var = f["imag"][()]

    real_var = np.transpose(real_var, (4, 3, 2, 1, 0))
    imag_var = np.transpose(imag_var, (4, 3, 2, 1, 0))

    # Proper magnitude variance approximation
    return real_var + imag_var  # (W,H,D,C,T)


def load_rsos(path):
    with h5py.File(path, "r") as f:
        img = f["img"][()]
    return np.transpose(img, (3, 2, 1, 0))  # (W,H,D,T)

def load_predicted_ocmr(recon_path):
    data = np.load(recon_path)
    image = data["image"]
    rsos = data["rsos"]
    return image, rsos

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def visualize_and_save(
    vol_orig, vol_pred,
    rsos_orig, rsos_pred,
    t_idx, d_idx, c_idx,
    save_path
):

    # rsos_orig_img = rsos_orig[:, :, d_idx, t_idx] * 100
    # vol_black_threshold = 1e-8
    # vol_non_black_mask = vol_orig_img > vol_black_threshold

    # -------------------------
    # SINGLE COIL VOLUME
    # -------------------------

    vol_orig_img = np.abs(vol_orig[:, :, d_idx, c_idx, t_idx])
    vol_pred_img = np.abs(vol_pred[:, :, d_idx, c_idx, t_idx])

    vol_orig_img = (vol_orig_img - vol_orig_img.min()) / (vol_orig_img.max() - vol_orig_img.min())
    vol_pred_img = (vol_pred_img - vol_pred_img.min()) / (vol_pred_img.max() - vol_pred_img.min())

    # Non-black mask for single-coil slice (based on original magnitude)
    vol_black_threshold = 1e-6
    vol_non_black_mask = vol_orig_img > vol_black_threshold

    vol_residual = vol_orig_img - vol_pred_img
    vol_residual = np.where(vol_non_black_mask, vol_residual, 0.0)

    # -------------------------
    # RSOS
    # -------------------------

    rsos_orig_img = rsos_orig[:, :, d_idx, t_idx]
    rsos_pred_img = rsos_pred[:, :, d_idx, t_idx]

    rsos_orig_img = (rsos_orig_img - rsos_orig_img.min()) / (rsos_orig_img.max() - rsos_orig_img.min())
    rsos_pred_img = (rsos_pred_img - rsos_pred_img.min()) / (rsos_pred_img.max() - rsos_pred_img.min())

    # Non-black mask for RSOS slice
    rsos_black_threshold = 1e-6
    rsos_non_black_mask = rsos_orig_img > rsos_black_threshold

    rsos_residual = rsos_orig_img - rsos_pred_img
    rsos_residual = np.where(rsos_non_black_mask, rsos_residual, 0.0)

    # -------------------------
    # RRMSE (on non-black indices only)
    # -------------------------

    vol_rrmse = compute_rrmse(vol_orig_img, vol_pred_img, mask=vol_non_black_mask)
    rsos_rrmse = compute_rrmse(rsos_orig_img, rsos_pred_img, mask=rsos_non_black_mask)

    print(f"Coil {c_idx} Volume RRMSE : {vol_rrmse:.4f}")
    print(f"RSOS RRMSE                : {rsos_rrmse:.4f}")

    # -------------------------

    vol_orig_img = vol_orig_img[75:-10, 0:-10]
    vol_pred_img = vol_pred_img[75:-10, 0:-10]
    vol_residual = vol_residual[75:-10, 0:-10]
    vmin_vol = np.percentile(vol_orig_img, 0)
    vmax_vol = np.percentile(vol_orig_img, 96)

    rsos_orig_img = rsos_orig_img[75:-10, 0:-10]
    rsos_pred_img = rsos_pred_img[75:-10, 0:-10]
    rsos_residual = rsos_residual[75:-10, 0:-10]
    vmin_rsos = np.percentile(rsos_orig_img, 0)
    vmax_rsos = np.percentile(rsos_orig_img, 96)

    # Plot
    # -------------------------

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # ---- Row 1: Volume ----
    axes[0,0].imshow(vol_orig_img.T, cmap='gray', vmin=vmin_vol, vmax=vmax_vol)
    axes[0,0].set_title(f"Volume Original")

    axes[0,1].imshow(vol_pred_img.T, cmap='gray', vmin=vmin_vol, vmax=vmax_vol)
    axes[0,1].set_title("Volume Predicted")

    im_vr = axes[0,2].imshow(vol_residual.T, cmap='seismic', vmin=-0.25, vmax=0.25)
    axes[0,2].set_title("Volume Residual")
    plt.colorbar(im_vr, ax=axes[0,2], orientation='horizontal')

    # ---- Row 2: RSOS ----
    axes[1, 0].imshow(rsos_orig_img.T, cmap="gray", vmin=vmin_rsos, vmax=vmax_rsos)
    axes[1,0].set_title("RSOS Original")

    axes[1, 1].imshow(rsos_pred_img.T, cmap="gray", vmin=vmin_rsos, vmax=vmax_rsos)
    axes[1,1].set_title("RSOS Predicted")

    im_rr = axes[1,2].imshow(rsos_residual.T, cmap='seismic', vmin=-0.25, vmax=0.25)
    axes[1,2].set_title("RSOS Residual")
    plt.colorbar(im_rr, ax=axes[1,2], orientation='horizontal')
    for ax in axes.flat:
        ax.axis("off")

    # Add RRMSE text to the plots
    axes[0, 1].text(
        0.5,
        -0.15,
        f"RRMSE: {vol_rrmse:.4f}",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="black",
    )
    axes[1, 1].text(
        0.5,
        -0.15,
        f"RRMSE: {rsos_rrmse:.4f}",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="black",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure to: {save_path}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    file_name = 'fs_0045_3T'
    original_path = f"../../../dataset/OCMR_data/{file_name}.h5"

    # compression_name = "ultra_compression"
    for compression_name in ["ultra_compression_N"]:#, "ultra_compression", "high_compression"]:
        recon_path = f"outputs/{compression_name}/{file_name}.nii.npz"

        output_folder = f"outputs/{compression_name}/plots"
        os.makedirs(output_folder, exist_ok=True)

        t_idx = 15
        d_idx = 4
        c_idx = 2   # choose coil
        file_sufix = f"t{t_idx}_d{d_idx}_c{c_idx}"

        print("Loading data...")
        vol_orig, rsos_orig = load_original_ocmr(original_path)
        vol_pred, rsos_pred = load_predicted_ocmr(recon_path)        

        visualize_and_save(
            vol_orig,
            vol_pred,
            rsos_orig,
            rsos_pred,
            t_idx,
            d_idx,
            c_idx,
            save_path=f"{output_folder}/comparison_{file_sufix}.png",
        )
