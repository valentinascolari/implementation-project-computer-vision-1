import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, filters
from skimage.segmentation import watershed, mark_boundaries
import cv2 as cv
from pathlib import Path


#  Global loading of the images
BASE = Path(".")

lena_orig = io.imread(BASE / "lena.tif")
lena_n10  = io.imread(BASE / "lena_noise_10.tif")
lena_n20  = io.imread(BASE / "lena_noise_20.tif")


#   QUESTION 2 — Gaussian filter per channel
def gaussian_rgb(img, sigma):
    imgf = img_as_float(img)
    k = int(np.ceil(6*sigma))
    if k % 2 == 0:
        k += 1
    out = np.zeros_like(imgf)
    for c in range(imgf.shape[2]):
        out[..., c] = cv.GaussianBlur(
            imgf[..., c], (k, k), sigmaX=sigma, sigmaY=sigma,
            borderType=cv.BORDER_REFLECT101
        )
    return np.clip(out, 0, 1)

def show_grid(img, title, sigmas=(0.8, 1.6, 2.4)):
    cols = 1 + len(sigmas)
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
    for j, s in enumerate(sigmas, start=1):
        filt = gaussian_rgb(img, s)
        axes[j].imshow(filt); axes[j].set_title(f"Gaussian σ={s}"); axes[j].axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.show()

def show_zoom(img, title, sigmas=(0.8, 1.6, 2.4)):
    H, W = img.shape[:2]
    h, w = int(H*0.35), int(W*0.35)
    r0, c0 = (H-h)//2, (W-w)//2
    crops = [img[r0:r0+h, c0:c0+w]]
    for s in sigmas:
        crops.append(gaussian_rgb(img, s)[r0:r0+h, c0:c0+w])
    cols = len(crops)
    fig, axes = plt.subplots(1, cols, figsize=(3.2*cols, 3.2))
    titles = ["Input"] + [f"σ={s}" for s in sigmas]
    for ax, im, t in zip(axes, crops, titles):
        ax.imshow(im); ax.set_title(t); ax.axis("off")
    fig.suptitle(f"Zoom — {title}"); plt.tight_layout(); plt.show()

def run_q2():
    print("\n>>> Running Question 2 — Gaussian Filter\n")
    show_grid(lena_orig, "Lena ORIGINAL (Gaussian per channel)")
    show_zoom(lena_orig, "Lena ORIGINAL")

    show_grid(lena_n10, "Lena noise_10 (Gaussian per channel)")
    show_zoom(lena_n10, "Lena noise_10")

    show_grid(lena_n20, "Lena noise_20 (Gaussian per channel)")
    show_zoom(lena_n20, "Lena noise_20")


#   QUESTION 3 — RGB Sobel
def sobel_edge_map_rgb(img_rgb):
    f = img_as_float(img_rgb)
    M2 = np.zeros(f.shape[:2], dtype=np.float64)
    for c in range(3):
        gx = filters.sobel_h(f[..., c])
        gy = filters.sobel_v(f[..., c])
        M2 += gx**2 + gy**2
    M = np.sqrt(M2 / 3.0)
    return (M - M.min()) / (np.ptp(M) + 1e-12)

def show_pair(img, M, title):
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].imshow(img); ax[0].set_title(f"{title} — input"); ax[0].axis("off")
    ax[1].imshow(M, cmap="gray"); ax[1].set_title(f"{title} — edge map"); ax[1].axis("off")
    plt.tight_layout(); plt.show()

def run_q3():
    print("\n>>> Running Question 3 — RGB Sobel\n")
    M_orig = sobel_edge_map_rgb(lena_orig)
    M_n10  = sobel_edge_map_rgb(lena_n10)
    M_n20  = sobel_edge_map_rgb(lena_n20)

    show_pair(lena_orig, M_orig, "Lena original")
    show_pair(lena_n10,  M_n10,  "Lena noise_10")
    show_pair(lena_n20,  M_n20,  "Lena noise_20")


#   QUESTION 4 — Thresholding
def threshold_maps(M, r):
    T_a = r * np.max(M)
    M_a = M.copy()
    M_a[M < T_a] = 0

    T_b = np.quantile(M, r)
    M_b = M.copy()
    M_b[M < T_b] = 0

    return M_a, M_b, T_a, T_b

def run_q4():
    print("\n>>> Running Question 4 — Thresholding\n")

    img_in = lena_n10
    sigma_g = 0.8

    img_filtered = gaussian_rgb(img_in, sigma_g)
    M = sobel_edge_map_rgb(img_filtered)

    r_val = 0.3
    M_a, M_b, T_a, T_b = threshold_maps(M, r_val)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_filtered); axes[0].set_title(f"I' (Gaussian σ={sigma_g})"); axes[0].axis("off")
    axes[1].imshow(M, cmap="gray"); axes[1].set_title("Map M"); axes[1].axis("off")
    axes[2].imshow(M_a, cmap="gray"); axes[2].set_title(f"M'_a — T={T_a:.3f}"); axes[2].axis("off")
    axes[3].imshow(M_b, cmap="gray"); axes[3].set_title(f"M'_b — T={T_b:.3f}"); axes[3].axis("off")
    plt.tight_layout(); plt.show()



#   QUESTION 5 — Watershed
def run_watershed(M_prime, img_orig_color):
    labels = watershed(M_prime, markers=None, mask=None)
    num_regions = np.max(labels)

    vis_boundaries = mark_boundaries(
        img_as_float(img_orig_color),
        labels,
        color=(1, 0, 0),
        mode="outer"
    )
    return labels, num_regions, vis_boundaries

def run_q5():
    print("\n>>> Running Question 5 — Watershed\n")

    sigma_g = 0.8
    r_val = 0.3

    img_filt = gaussian_rgb(lena_n10, sigma_g)
    M = sobel_edge_map_rgb(img_filt)
    M_a, M_b, *_ = threshold_maps(M, r_val)

    labels_base, regions_base, vis_base = run_watershed(M, img_filt)
    labels_a, regions_a, vis_a = run_watershed(M_a, img_filt)
    labels_b, regions_b, vis_b = run_watershed(M_b, img_filt)

    print(f"Baseline: {regions_base} regions")
    print(f"Threshold (a): {regions_a} regions")
    print(f"Threshold (b): {regions_b} regions\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(vis_base); axes[0].set_title(f"Baseline: {regions_base} regions"); axes[0].axis("off")
    axes[1].imshow(vis_a); axes[1].set_title(f"Q4a: {regions_a} regions"); axes[1].axis("off")
    axes[2].imshow(vis_b); axes[2].set_title(f"Q4b: {regions_b} regions"); axes[2].axis("off")
    plt.tight_layout(); plt.show()


#   MAIN MENU
def menu():
    while True:
        print("\n==============================")
        print("         MAIN MENU")
        print("==============================")
        print("1 — Question 2 (Gaussian)")
        print("2 — Question 3 (Sobel)")
        print("3 — Question 4 (Thresholding)")
        print("4 — Question 5 (Watershed)")
        print("5 — Run all")
        print("0 — Exit")

        op = input("\nChoose: ").strip()

        if op == "1": run_q2()
        elif op == "2": run_q3()
        elif op == "3": run_q4()
        elif op == "4": run_q5()
        elif op == "5":
            run_q2(); run_q3(); run_q4(); run_q5()
        elif op == "0":
            print("Exiting…")
            break
        else:
            print("Invalid option!")

if __name__ == "__main__":
    menu()