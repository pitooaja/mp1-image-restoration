# Mini Project 1 — Image Restoration

**Mata Kuliah:** Pengolahan Citra dan Video  
**Nama:** Fito Dwi Ardiansah  
**NRP:** 5024241053

---

## Deskripsi

Proyek ini bertujuan merestorasi citra Lena yang telah mengalami empat jenis kerusakan sekaligus: low contrast, Gaussian noise, salt-and-pepper noise, dan blur. Restorasi dilakukan menggunakan pipeline 6 langkah yang semuanya diimplementasikan manual dengan NumPy. OpenCV hanya digunakan untuk membaca dan menyimpan citra.

---

## Pipeline Restorasi

```
Input (Noisy)
     |
     v
Step 1: Median Filter (3x3)            <- hapus salt-and-pepper noise
     |
     v
Step 2: Gaussian Filter (5x5, s=1.2)   <- reduksi Gaussian noise
     |
     v
Step 3: Wiener Filter (FFT)             <- koreksi blur (frequency domain)
     |
     v
Step 4: Histogram Equalization          <- perbaiki kontras rendah
     |
     v
Step 5: Laplacian Sharpening (a=0.7)   <- pertajam tepi
     |
     v
Step 6: Unsharp Masking (amount=1.5)   <- pertajam detail halus
     |
     v
Output (Restored)
```

### Penjelasan Tiap Langkah

**Step 1 — Median Filter**  
Digunakan untuk menghilangkan salt-and-pepper noise. Cara kerjanya: setiap piksel diganti dengan nilai median dari piksel-piksel tetangganya dalam window 3x3. Median dipilih karena tidak terpengaruh nilai ekstrem (piksel hitam/putih acak), berbeda dengan mean filter yang justru akan menyebarkan noise tersebut.

**Step 2 — Gaussian Filter**  
Setelah S&P noise hilang, masih ada Gaussian noise yang tersebar. Gaussian filter menghaluskan sisa noise ini dengan konvolusi menggunakan kernel berbobot Gaussian — piksel yang lebih dekat ke pusat kernel diberi bobot lebih besar. Hasilnya lebih natural dibanding rata-rata biasa.

**Step 3 — Wiener Filter**  
Digunakan untuk mengatasi blur menggunakan pendekatan frequency domain (FFT). Citra diubah ke domain frekuensi, lalu filter Wiener diterapkan dengan rumus `H* / (|H|^2 + K)` untuk membalik efek blur. Nilai K berfungsi sebagai regularisasi agar tidak terlalu sensitif terhadap noise. Catatan: efektivitasnya bergantung pada seberapa akurat asumsi PSF (model blur) yang digunakan.

**Step 4 — Histogram Equalization**  
Dilakukan setelah denoising selesai agar noise tidak ikut diperkuat. HE menyebarkan distribusi intensitas yang awalnya sempit (menumpuk di tengah) ke seluruh range 0-255 menggunakan pemetaan berbasis CDF. Hasilnya kontras meningkat signifikan.

**Step 5 — Laplacian Sharpening**  
Kernel Laplacian mendeteksi perubahan intensitas tajam (tepi). Hasil deteksi tepi ini dikurangkan dari gambar asli sehingga transisi antar region menjadi lebih tajam. Parameter alpha=0.7 dipilih agar efeknya tidak terlalu agresif dan menimbulkan artefak.

**Step 6 — Unsharp Masking**  
Langkah terakhir untuk mempertajam detail halus yang mungkin masih terlihat lunak. Cara kerjanya: gambar di-blur dulu, lalu selisih antara gambar asli dan versi blur-nya (yang berisi detail) ditambahkan kembali ke gambar asli dengan faktor pengali 1.5.

### Alasan Urutan

Urutan pipeline ini disusun berdasarkan prinsip: selesaikan masalah yang paling merusak terlebih dahulu sebelum melakukan enhancement.

- Denoising dilakukan paling awal karena noise yang masih ada akan ikut diperkuat oleh proses sharpening dan HE jika dilakukan setelahnya.
- Median sebelum Gaussian karena median lebih efisien untuk outlier ekstrem (piksel putih/hitam), sehingga Gaussian tidak perlu bekerja keras.
- HE setelah deblur agar rentang intensitas sudah stabil sebelum disebarkan.
- Sharpening di akhir agar yang dipertajam adalah sinyal gambar, bukan noise.

---

## Perbandingan Visual

Lihat file `output/pipeline_visualization.png` untuk melihat perubahan citra dan histogram di setiap tahap.

### Statistik Intensitas

| Stage | Min | Max | Mean | Std |
|---|---|---|---|---|
| Noisy (Input) | 0 | ~140 | ~78 | ~44 |
| After Median Filter | 0 | ~140 | ~78 | ~44 |
| After Gaussian Filter | 0 | ~138 | ~77 | ~44 |
| After Wiener Deblur | 0 | ~230 | ~76 | ~44 |
| After Hist. Eq. | 0 | 255 | ~121 | ~77 |
| After Laplacian | 0 | 255 | ~120 | ~76 |
| Restored (Final) | 0 | 255 | ~121 | ~77 |

---

## Analisis Hasil

**Yang berhasil:**
- Salt-and-pepper noise hilang hampir sepenuhnya setelah median filter
- Kontras meningkat drastis setelah histogram equalization — range intensitas melebar dari sekitar 140 menjadi 255
- Tepi dan detail gambar lebih tajam setelah Laplacian + Unsharp Masking

**Keterbatasan:**
- Wiener Filter kurang optimal karena PSF (model blur) yang digunakan hanya asumsi — parameter sigma_blur dan K idealnya dikalibrasi berdasarkan karakteristik blur aslinya
- Histogram Equalization global kadang membuat beberapa area terlihat terlalu terang atau kontras tidak merata; CLAHE (adaptive) bisa menjadi alternatif yang lebih baik
- Laplacian sharpening sensitif terhadap sisa noise — jika noise belum bersih sempurna, artefak bisa muncul di area halus

---

## Cara Menjalankan

**Install dependencies:**
```bash
pip install numpy opencv-python matplotlib
```

**Struktur folder:**
```
mp1-image-restoration/
├── README.md
├── restoration.py
├── input/
│   └── lena_noisy.png
└── output/
    ├── lena_restored.png
    └── pipeline_visualization.png
```

**Jalankan:**
```bash
cd mp1-image-restoration
python restoration.py
```

Program akan membaca `input/lena_noisy.png`, menjalankan 6 tahap restorasi, menyimpan hasil ke folder `output/`, dan menampilkan visualisasi pipeline beserta histogram tiap tahap.

---

## Referensi

- Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*, 4th ed.
- Materi kuliah Pengolahan Citra dan Video, Pertemuan 1-7
