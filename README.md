# Mini Project 1  Image Restoration

**Mata Kuliah:** Pengolahan Citra dan Video  
**Nama:** Fito Dwi Ardiansah  
**NRP:** 5024241053

---

## Deskripsi

Proyek ini bertujuan merestorasi citra Lena yang telah mengalami empat jenis kerusakan sekaligus: low contrast, Gaussian noise, salt-and-pepper noise, dan blur. Restorasi dilakukan menggunakan pipeline 4 langkah yang semuanya diimplementasikan manual dengan NumPy. OpenCV hanya digunakan untuk membaca dan menyimpan citra.

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
Step 3: Histogram Equalization          <- perbaiki kontras rendah
     |
     v
Step 4: Unsharp Masking (amount=1.5)   <- pertajam detail yang kabur
     |
     v
Output (Restored)
```

### Penjelasan Tiap Langkah

**Step 1 Median Filter**  
Digunakan untuk menghilangkan salt-and-pepper noise. Cara kerjanya: setiap piksel diganti dengan nilai median dari piksel-piksel tetangganya dalam window 3x3. Median dipilih karena tidak terpengaruh nilai ekstrem (piksel hitam/putih acak), berbeda dengan mean filter yang justru akan menyebarkan noise tersebut.

**Step 2 Gaussian Filter**  
Setelah S&P noise hilang, masih ada Gaussian noise yang tersebar. Gaussian filter menghaluskan sisa noise ini dengan konvolusi menggunakan kernel berbobot Gaussian piksel yang lebih dekat ke pusat kernel diberi bobot lebih besar. Hasilnya lebih natural dibanding rata-rata biasa.

**Step 3 Histogram Equalization**  
Dilakukan setelah denoising selesai agar noise tidak ikut diperkuat. HE menyebarkan distribusi intensitas yang awalnya sempit (menumpuk di tengah) ke seluruh range 0-255 menggunakan pemetaan berbasis CDF. Hasilnya kontras meningkat signifikan.

**Step 4 Unsharp Masking**  
Langkah terakhir untuk mempertajam detail dan tepi yang terlihat lunak akibat proses filtering sebelumnya. Cara kerjanya: gambar di-blur dulu, lalu selisih antara gambar asli dan versi blur-nya (yang berisi detail) ditambahkan kembali ke gambar asli dengan faktor pengali 1.5.

### Alasan Urutan

- Denoising dilakukan paling awal karena noise yang masih ada akan ikut diperkuat oleh HE dan sharpening jika dilakukan setelahnya.
- Median sebelum Gaussian karena median lebih efisien untuk outlier ekstrem (piksel putih/hitam), sehingga Gaussian tidak perlu bekerja keras.
- HE setelah denoising agar distribusi intensitas sudah stabil sebelum disebarkan ke range penuh.
- Unsharp masking di akhir agar yang dipertajam adalah sinyal gambar, bukan noise.

---

## Perbandingan Visual

Lihat file `output/pipeline_visualization.png` untuk melihat perubahan citra dan histogram di setiap tahap.

### Statistik Intensitas

| Stage | Min | Max | Mean | Std |
|---|---|---|---|---|
| Noisy (Input) | 0 | ~140 | ~78 | ~44 |
| After Median Filter | 0 | ~140 | ~78 | ~44 |
| After Gaussian Filter | 0 | ~138 | ~77 | ~44 |
| After Hist. Eq. | 0 | 255 | ~121 | ~77 |
| Restored (Final) | 0 | 255 | ~121 | ~77 |

---

## Analisis Hasil

**Yang berhasil:**
- Salt-and-pepper noise hilang hampir sepenuhnya setelah median filter
- Kontras meningkat drastis setelah histogram equalization range intensitas melebar dari sekitar 140 menjadi 255
- Detail dan tepi gambar lebih tajam setelah unsharp masking

**Keterbatasan:**
- Gaussian filter menyebabkan sedikit blur tambahan bilateral filter bisa menjadi alternatif yang lebih baik karena mempertahankan tepi
- Histogram equalization global kadang membuat beberapa area terlihat terlalu terang; CLAHE (adaptive) bisa memberikan hasil yang lebih merata
- Parameter amount pada unsharp masking perlu dikalibrasi
- nilai terlalu tinggi bisa menimbulkan halo artifact di sekitar tepi

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

Program akan membaca `input/lena_noisy.png`, menjalankan 4 tahap restorasi, menyimpan hasil ke folder `output/`, dan menampilkan visualisasi pipeline beserta histogram tiap tahap.

---

## Referensi

- Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*, 4th ed.
- Materi kuliah Pengolahan Citra dan Video, Pertemuan 1-7
