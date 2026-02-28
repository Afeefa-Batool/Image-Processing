# Install necessary libraries
!pip install opencv-python-headless

import cv2
import numpy as np
from google.colab import files
from IPython.display import Image, display

# Function to enhance image
def enhance_image(image_path, alpha=1.5, beta=50):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please make sure the file exists.")
        return None

    # Adjust brightness and contrast
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply Gaussian blur for noise reduction
    denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    return denoised_image

# Upload image
uploaded = files.upload()
for fn in uploaded.keys():
    image_path = fn

# Process the image
processed_image = enhance_image(image_path)

# Save the processed image
cv2.imwrite('processed_image.jpg', processed_image)

# Display the original and processed images
print(f"Displaying original image: {image_path}")
display(Image(filename=image_path))
print(f"Displaying processed image: processed_image.jpg")
display(Image(filename='processed_image.jpg'))

# Provide download link
files.download('processed_image.jpg')