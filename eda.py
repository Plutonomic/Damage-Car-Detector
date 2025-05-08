# --- eda.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def analyze_image_dimensions(image):
    fig, ax = plt.subplots()
    height, width = image.shape[:2]
    ax.bar(["Width", "Height"], [width, height], color=["skyblue", "salmon"])
    ax.set_title("Image Dimensions")
    return fig

def plot_rgb_histogram(image):
    fig, ax = plt.subplots()
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_title("RGB Color Distribution")
    return fig
