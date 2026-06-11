from read_ocmr import read_ocmr
import numpy as np
import os

OCMR_MAT_DATA_DIR = "../../dataset/OCMR_data/"

def kspace_to_image_2d(kspace):
    """
    2D IFFT over kx, ky with proper shifts (works on any trailing dims).

    Args:
        kspace: np.ndarray, complex, shape (kx, ky, ...)
    Returns:
        image:  np.ndarray, complex, same shape
    """
    image = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=[0, 1]),
            axes=[0, 1],
        ),
        axes=[0, 1],
    )
    return image


def load_vol(file_path):
    # 1. Construct Paths
    h5_path = os.path.join(file_path)
    # 2. Load k-space via read_ocmr
    #    kData shape: (kx, ky, kz, coil, phase, set, slice, rep, avg)
    try:
        kData, param = read_ocmr(h5_path)
        print(f"  Raw kData shape: {kData.shape}")
        print(f"  Dim order      : {param['kspace_dim']}")
    except Exception as e:
        print(f"Error loading OCMR file: {e}")
        return

    # 3. Average over avg dim (axis 8)
    kData_tmp = np.mean(kData, axis=8)  # → (kx, ky, kz, coil, phase, set, slice, rep)

    # 4. 2D IFFT over kx, ky (axes 0, 1)
    print("  Applying 2D IFFT over (kx, ky)...")
    im_coil = kspace_to_image_2d(
        kData_tmp
    )  # → (kx, ky, kz, coil, phase, set, slice, rep)

    # 5. Retain only (kx, ky, slice, coil, phase) — remove kz, set, rep
    #    kz: take index 0 (squeeze singleton), set: index 0, rep: index 0
    im_coil = im_coil[:, :, 0, :, :, 0, :, 0]  # → (kx, ky, coil, phase, slice)
    #    Reorder to (kx, ky, slice, coil, phase)
    im_coil = np.transpose(im_coil, (0, 1, 4, 2, 3))  # → (kx, ky, slice, coil, phase)
    image_complex = im_coil.astype(np.complex64)

    W, H, D = (
        image_complex.shape[0],
        image_complex.shape[1],
        image_complex.shape[2],
    )  # kx, ky, slice
    num_coils = image_complex.shape[3]
    T = image_complex.shape[4]  # number of phases
    D_out = num_coils * T * 2
    print(
        f"  Image Shape: ({W}, {H}, {D}) spatial, {num_coils} coils, {T} phases → D_out = {D_out}"
    )

    # 6. Split into real/imag → (W, H, D, C, T, 2)
    image_ri = np.stack([image_complex.real, image_complex.imag], axis=-1).astype(
        np.float32
    )

    return image_ri
