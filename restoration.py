"""
=============================================================
  Mini Project 1 — Image Restoration
  Mata Kuliah : Pengolahan Citra dan Video
=============================================================
  Nama  : Fito Dwi Ardiansah
  NRP   : 5024241053
=============================================================

Pipeline Restorasi (diproses per-channel R, G, B):
  1. Median Filter  (5x5)          -> Hapus salt-and-pepper noise
  2. Gaussian Filter (7x7, s=1.5)  -> Reduksi Gaussian noise
  3. Histogram Equalization         -> Perbaiki kontras rendah
  4. Unsharp Masking (amount=0.7)  -> Pertajam detail yang kabur

Semua operasi diimplementasikan MANUAL menggunakan NumPy.
OpenCV hanya digunakan untuk cv2.imread dan cv2.imwrite.
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
#  Helper: Zero-Padding
# ─────────────────────────────────────────────
def pad_image(img, pad):
    """Tambahkan zero-padding sebesar `pad` piksel di setiap sisi."""
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
    padded[pad:pad + h, pad:pad + w] = img
    return padded


# ─────────────────────────────────────────────
#  Step 1 — Median Filter (Manual)
#  Efektif untuk menghilangkan salt-and-pepper
# ─────────────────────────────────────────────
def median_filter(img, kernel_size=5):
    """Median filter manual. Setiap piksel diganti median tetangganya."""
    assert kernel_size % 2 == 1, "kernel_size harus ganjil"
    pad    = kernel_size // 2
    padded = pad_image(img.astype(np.float64), pad)
    h, w   = img.shape
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(patch)

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Step 2 — Gaussian Filter (Manual)
#  Mereduksi Gaussian noise dengan smoothing
# ─────────────────────────────────────────────
def make_gaussian_kernel(size, sigma):
    """Buat kernel Gaussian 2D berukuran (size x size)."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_filter(img, kernel_size=7, sigma=1.5):
    """Konvolusi manual dengan kernel Gaussian."""
    kernel = make_gaussian_kernel(kernel_size, sigma)
    pad    = kernel_size // 2
    padded = pad_image(img.astype(np.float64), pad)
    h, w   = img.shape
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(patch * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Step 3 — Histogram Equalization (Manual)
#  Memperbaiki kontras rendah
# ─────────────────────────────────────────────
def histogram_equalization(img):
    """HE manual berbasis CDF — input/output per-channel (2D)."""
    h, w  = img.shape
    total = h * w

    # Hitung histogram
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1

    # CDF
    cdf     = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()

    # Look-Up Table
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
    return lut[img]


# ─────────────────────────────────────────────
#  Step 4 — Unsharp Masking (Manual)
#  Mempertajam detail dan tepi yang kabur
# ─────────────────────────────────────────────
def unsharp_masking(img, blur_size=5, sigma=1.0, amount=0.7):
    """
    Unsharp Masking:  sharpened = original + amount * (original - blurred)
    Bisa menerima 2D (1 channel) atau 3D (RGB).
    """
    if img.ndim == 3:
        # Proses tiap channel terpisah
        channels = []
        for c in range(3):
            ch = img[:, :, c]
            blurred = gaussian_filter(ch, blur_size, sigma).astype(np.float64)
            ch_f    = ch.astype(np.float64)
            sharp   = ch_f + amount * (ch_f - blurred)
            channels.append(np.clip(sharp, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=2)
    else:
        blurred = gaussian_filter(img, blur_size, sigma).astype(np.float64)
        img_f   = img.astype(np.float64)
        sharp   = img_f + amount * (img_f - blurred)
        return np.clip(sharp, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Helper: Compute Histogram (untuk visualisasi)
# ─────────────────────────────────────────────
def compute_histogram(img):
    """Hitung histogram dari citra grayscale (2D)."""
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1
    return hist


# ─────────────────────────────────────────────
#  Helper: RGB -> Grayscale (untuk visualisasi)
# ─────────────────────────────────────────────
def rgb_to_gray(img):
    return (0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2]).astype(np.uint8)


# ─────────────────────────────────────────────
#  Helper: Proses satu channel (full pipeline)
# ─────────────────────────────────────────────
def process_channel(ch):
    """Jalankan Step 1-3 pada satu channel grayscale."""
    s1 = median_filter(ch, kernel_size=5)
    s2 = gaussian_filter(s1, kernel_size=7, sigma=1.5)
    s3 = histogram_equalization(s2)
    return s1, s2, s3


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────
def restore():
    print("=" * 60)
    print("  Mini Project 1 -- Image Restoration Pipeline")
    print("=" * 60)

    # ── Load citra ──
    img_bgr = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {INPUT_PATH}")
    img_noisy = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"[OK] Citra dimuat: {INPUT_PATH}  {img_noisy.shape}")

    # ── Proses tiap channel R, G, B ──
    print("\nMemproses channel R, G, B ...\n")
    results = {"step1": [], "step2": [], "step3": []}

    for c, name in enumerate(["R", "G", "B"]):
        print(f"  Channel {name}:")
        ch = img_noisy[:, :, c]
        s1, s2, s3 = process_channel(ch)
        results["step1"].append(s1)
        results["step2"].append(s2)
        results["step3"].append(s3)
        print(f"    [1/3] Median Filter       -- selesai")
        print(f"    [2/3] Gaussian Filter     -- selesai")
        print(f"    [3/3] Histogram EQ        -- selesai")

    # ── Gabungkan channel ──
    step1_rgb = np.stack(results["step1"], axis=2)
    step2_rgb = np.stack(results["step2"], axis=2)
    step3_rgb = np.stack(results["step3"], axis=2)

    # ── Step 4: Unsharp Masking (pada citra RGB gabungan) ──
    print("\n  Sharpening:")
    restored = unsharp_masking(step3_rgb, blur_size=5, sigma=1.0, amount=0.7)
    print(f"    [4/4] Unsharp Masking     -- selesai")

    # ── Simpan hasil ──
    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    print(f"\n[OK] Hasil disimpan: {OUTPUT_PATH}")

    # ─────────────────────────────────────────
    #  Visualisasi Pipeline (5 tahap + histogram)
    # ─────────────────────────────────────────
    stages = [
        ("Noisy (Input)",         img_noisy),
        ("After Median Filter",   step1_rgb),
        ("After Gaussian Filter", step2_rgb),
        ("After Hist. Eq.",       step3_rgb),
        ("Restored (Final)",      restored),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Image Restoration Pipeline", fontsize=16, fontweight="bold")

    for col, (title, img) in enumerate(stages):
        # Baris atas: citra
        axes[0, col].imshow(img)
        color = "black"
        if col == 0:
            color = "red"
        elif col == 4:
            color = "green"
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", color=color)
        axes[0, col].axis("off")

        # Baris bawah: histogram (grayscale average)
        gray = rgb_to_gray(img)
        hist = compute_histogram(gray)
        bar_color = "steelblue" if (col == 0 or col == 4) else "gray"
        axes[1, col].bar(range(256), hist, color=bar_color, width=1, alpha=0.85)
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("Intensitas", fontsize=8)
        axes[1, col].set_ylabel("Frekuensi", fontsize=8)
        axes[1, col].tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("output/pipeline_visualization.png", dpi=150, bbox_inches="tight")
    print("[OK] Visualisasi pipeline disimpan: output/pipeline_visualization.png")

    # ─────────────────────────────────────────
    #  Visualisasi Before vs After
    # ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.suptitle("Before vs After Restoration", fontsize=14, fontweight="bold")

    axes2[0].imshow(img_noisy)
    axes2[0].set_title("Before (Noisy Input)", fontsize=12, fontweight="bold", color="red")
    axes2[0].axis("off")

    axes2[1].imshow(restored)
    axes2[1].set_title("After (Restored)", fontsize=12, fontweight="bold", color="green")
    axes2[1].axis("off")

    plt.tight_layout()
    plt.savefig("output/before_after.png", dpi=150, bbox_inches="tight")
    print("[OK] Before/After disimpan: output/before_after.png")

    plt.show()

    # ─────────────────────────────────────────
    #  Statistik Intensitas
    # ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Stage':<25} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7}")
    print(f"  {'-'*53}")
    for title, img in stages:
        flat = img.astype(np.float64)
        print(f"  {title:<25} {flat.min():>5.0f} {flat.max():>5.0f} "
              f"{flat.mean():>7.1f} {flat.std():>7.1f}")

    print(f"\n[OK] Pipeline selesai!")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    restore()