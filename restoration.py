"""
=============================================================
  Mini Project 1 — Image Restoration
  Mata Kuliah: Pengolahan Citra dan Video
=============================================================
  Nama : Fito Dwi Ardiansah
  NRP  : 5024241053
=============================================================

Pipeline Restorasi:
  1. Median Filter (manual)          → Menghilangkan salt-and-pepper noise
  2. Gaussian Filter (manual)        → Mereduksi Gaussian noise (smoothing)
  3. Histogram Equalization (manual) → Memperbaiki kontras rendah
  4. Unsharp Masking (manual)        → Mempertajam detail yang kabur

Semua operasi dilakukan MANUAL menggunakan NumPy.
OpenCV hanya digunakan untuk imread dan imwrite.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
#  I/O Paths
# ─────────────────────────────────────────────
INPUT_PATH  = "input/lena_noisy.png"
OUTPUT_PATH = "output/lena_restored.png"
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────
#  Helper: pad image (zero-padding)
# ─────────────────────────────────────────────
def pad_image(img: np.ndarray, pad: int) -> np.ndarray:
    """Zero-pad image by `pad` pixels on each side."""
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
    padded[pad:pad + h, pad:pad + w] = img
    return padded


# ─────────────────────────────────────────────
#  Step 1 — Median Filter (Manual)
#  Sangat efektif untuk salt-and-pepper noise
# ─────────────────────────────────────────────
def median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Manual median filter.
    
    Args:
        img         : grayscale uint8 image
        kernel_size : odd integer (e.g. 3, 5, 7)
    
    Returns:
        filtered image (uint8)
    """
    assert kernel_size % 2 == 1, "kernel_size harus bilangan ganjil"
    pad  = kernel_size // 2
    img_f = img.astype(np.float64)
    padded = pad_image(img_f, pad)
    h, w   = img.shape
    output = np.zeros_like(img_f)

    for i in range(h):
        for j in range(w):
            patch   = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(patch)

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Step 2 — Gaussian Filter (Manual)
#  Mengurangi Gaussian noise dengan smoothing
# ─────────────────────────────────────────────
def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Buat kernel Gaussian 2D berukuran size×size."""
    ax   = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_filter(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Manual Gaussian blur menggunakan konvolusi 2D.
    
    Args:
        img         : grayscale uint8 image
        kernel_size : ukuran kernel (ganjil)
        sigma       : standar deviasi Gaussian
    
    Returns:
        blurred image (uint8)
    """
    kernel = make_gaussian_kernel(kernel_size, sigma)
    pad    = kernel_size // 2
    img_f  = img.astype(np.float64)
    padded = pad_image(img_f, pad)
    h, w   = img.shape
    output = np.zeros_like(img_f)

    for i in range(h):
        for j in range(w):
            patch       = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(patch * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Step 3 — Histogram Equalization (Manual)
#  Memperbaiki kontras rendah (low contrast)
# ─────────────────────────────────────────────
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Manual histogram equalization untuk grayscale image.
    
    Langkah:
      1. Hitung histogram (frekuensi setiap intensitas 0–255)
      2. Hitung CDF (Cumulative Distribution Function)
      3. Normalisasi CDF → lookup table [0, 255]
      4. Map setiap piksel lewat lookup table
    
    Returns:
        equalized image (uint8)
    """
    h, w  = img.shape
    total = h * w

    # 1. Histogram
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1

    # 2. CDF
    cdf = np.cumsum(hist)

    # 3. Normalisasi CDF → [0, 255]
    cdf_min = cdf[cdf > 0].min()
    lut = np.round(
        (cdf - cdf_min) / (total - cdf_min) * 255
    ).astype(np.uint8)

    # 4. Map piksel
    equalized = lut[img]
    return equalized


# ─────────────────────────────────────────────
#  Step 4 — Unsharp Masking (Manual)
#  Mempertajam tepi dan detail yang kabur
# ─────────────────────────────────────────────
def unsharp_masking(img: np.ndarray,
                    blur_size: int = 5,
                    sigma: float   = 1.0,
                    amount: float  = 1.5) -> np.ndarray:
    """
    Manual Unsharp Masking.
    
    Formula:  output = original + amount × (original − blurred)
                     = (1 + amount) × original − amount × blurred
    
    Args:
        img       : grayscale uint8 image
        blur_size : kernel size untuk Gaussian blur internal
        sigma     : sigma untuk Gaussian blur internal
        amount    : kekuatan sharpening (1.0–2.0 direkomendasikan)
    
    Returns:
        sharpened image (uint8)
    """
    img_f   = img.astype(np.float64)
    blurred = gaussian_filter(img, blur_size, sigma).astype(np.float64)
    mask    = img_f - blurred                        # detail mask
    sharp   = img_f + amount * mask                  # tambahkan detail
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Compute Histogram (untuk visualisasi)
# ─────────────────────────────────────────────
def compute_histogram(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1
    return hist


# ─────────────────────────────────────────────
#  Compute PSNR (opsional, jika ada referensi)
# ─────────────────────────────────────────────
def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Lebih tinggi = lebih baik."""
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────
def restore(input_path: str, output_path: str):
    print("=" * 60)
    print("  Mini Project 1 — Image Restoration Pipeline")
    print("=" * 60)

    # ── Load ──────────────────────────────────
    img_noisy = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_noisy is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {input_path}")
    print(f"[✓] Citra dimuat: {input_path}  {img_noisy.shape}")

    # ── Step 1: Median Filter ─────────────────
    print("\n[1/4] Median Filter (kernel 3×3) — menghilangkan salt-and-pepper ...")
    step1 = median_filter(img_noisy, kernel_size=3)
    print("      Selesai.")

    # ── Step 2: Gaussian Filter ───────────────
    print("[2/4] Gaussian Filter (kernel 5×5, σ=1.2) — mereduksi Gaussian noise ...")
    step2 = gaussian_filter(step1, kernel_size=5, sigma=1.2)
    print("      Selesai.")

    # ── Step 3: Histogram Equalization ───────
    print("[3/4] Histogram Equalization — memperbaiki kontras ...")
    step3 = histogram_equalization(step2)
    print("      Selesai.")

    # ── Step 4: Unsharp Masking ───────────────
    print("[4/4] Unsharp Masking (amount=1.5) — mempertajam detail ...")
    step4 = unsharp_masking(step3, blur_size=5, sigma=1.0, amount=1.5)
    print("      Selesai.")

    restored = step4

    # ── Save ──────────────────────────────────
    cv2.imwrite(output_path, restored)
    print(f"\n[✓] Hasil disimpan: {output_path}")

    # ── Visualisasi ───────────────────────────
    stages = {
        "Noisy (Input)"        : img_noisy,
        "After Median Filter"  : step1,
        "After Gaussian Filter": step2,
        "After Hist. Eq."      : step3,
        "Restored (Final)"     : restored,
    }

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Image Restoration Pipeline", fontsize=16, fontweight="bold")

    for col, (title, img) in enumerate(stages.items()):
        # Gambar citra
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # Histogram
        hist = compute_histogram(img)
        axes[1, col].bar(range(256), hist, color="steelblue", width=1, alpha=0.85)
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("Intensitas", fontsize=8)
        axes[1, col].set_ylabel("Frekuensi", fontsize=8)
        axes[1, col].tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("output/pipeline_visualization.png", dpi=150, bbox_inches="tight")
    print("[✓] Visualisasi disimpan: output/pipeline_visualization.png")
    plt.show()

    # ── Statistik ─────────────────────────────
    print("\n── Statistik Intensitas ─────────────────────")
    print(f"  {'Stage':<25} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7}")
    print(f"  {'-'*53}")
    for title, img in stages.items():
        print(f"  {title:<25} {img.min():>5} {img.max():>5} {img.mean():>7.1f} {img.std():>7.1f}")

    print("\n[✓] Pipeline selesai!")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    restore(INPUT_PATH, OUTPUT_PATH)