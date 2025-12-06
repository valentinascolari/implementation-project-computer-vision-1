# 📘 First Assignment — Computer Vision

This repository contains the implementation of the **first assignment** for the Computer Vision course.  
The project includes Gaussian filtering, Sobel edge detection, thresholding, and Watershed segmentation.  
All required images are included in the folder along with the source code.

---

## 📂 Repository Structure
```bash
/
├── first_assignment.py
├── lena.tif
├── lena_noise_10.tif
├── lena_noise_20.tif
├── macbeth.tif
├── macbeth_noise_10.tif
├── macbeth_noise_20.tif
├── peppers.tif
├── peppers_noise_10.tif
└── peppers_noise_20.tif
```

All images must remain in the **same directory** as the Python script so the code can load them automatically.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install numpy matplotlib opencv-python scikit-image
```
### 2. Run the Script
```bash
python first_assignment.py
```

##📋 Available Menu Options

When executing the script, the following menu will appear:

```bash
#MENU
1 — Question 2 (Gaussian Filter)
2 — Question 3 (Sobel RGB)
3 — Question 4 (Thresholding)
4 — Question 5 (Watershed)
5 — Run all
0 — Exit
```
Choose an option and the corresponding visualizations will open in Matplotlib.

## 📘 Detailed Analysis

All five questions were analyzed in detail, including visual results, comparisons, and discussions.  
You can access the full analysis in the Google Colab notebook below:

🔗 **Full analysis available here:**  
https://colab.research.google.com/drive/1MbMZh8KvtsAH0DtgExiv7ZlgztSJ0B2B?usp=sharing
