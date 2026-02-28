import gradio as gr
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

# -------------------------------
# Image Enhancement Methods
# -------------------------------

def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# -------------------------------
# Evaluation Metrics
# -------------------------------

def evaluate(original, processed):
    psnr = peak_signal_noise_ratio(original, processed)
    ssim = structural_similarity(
        cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    )
    return round(psnr, 2), round(ssim, 4)

# -------------------------------
# Main Processing Function
# -------------------------------

def process_image(input_image):
    image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    gamma_img = gamma_correction(image)
    clahe_img = clahe_enhancement(image)
    denoise_img = denoise_image(image)

    gamma_psnr, gamma_ssim = evaluate(image, gamma_img)
    clahe_psnr, clahe_ssim = evaluate(image, clahe_img)
    denoise_psnr, denoise_ssim = evaluate(image, denoise_img)

    results_text = f"""
    🔬 Image Enhancement Comparison

    Gamma Correction → PSNR: {gamma_psnr}, SSIM: {gamma_ssim}
    CLAHE Enhancement → PSNR: {clahe_psnr}, SSIM: {clahe_ssim}
    Denoising → PSNR: {denoise_psnr}, SSIM: {denoise_ssim}

    Higher SSIM indicates better structural preservation.
    """

    return (
        cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(denoise_img, cv2.COLOR_BGR2RGB),
        results_text
    )

# -------------------------------
# Gradio Interface
# -------------------------------

with gr.Blocks() as demo:
    gr.Markdown("""
    # Adaptive Low-Light Image Enhancement Lab

    This research-oriented demo compares classical enhancement techniques 
    using objective quality metrics (PSNR & SSIM).
    """)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")

    with gr.Row():
        gamma_output = gr.Image(label="Gamma Correction")
        clahe_output = gr.Image(label="CLAHE Enhancement")
        denoise_output = gr.Image(label="Denoised Output")

    metrics_output = gr.Textbox(label="Evaluation Metrics")

    btn = gr.Button("Run Enhancement & Evaluation")
    btn.click(process_image, 
              inputs=input_image,
              outputs=[gamma_output, clahe_output, denoise_output, metrics_output])

demo.launch()