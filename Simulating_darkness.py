import os
import numpy as np
from PIL import Image, ImageEnhance
import glob

# Set up the input and output folders
input_folder = r'FOD-A'
output_folder = r'FOD-B'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate random values for gamma, a, and b
gamma = np.random.uniform(0.5, 2.0)
a = np.random.uniform(0.5, 1)
b = np.random.uniform(0.5, 1)

# Find all image files in the input folder with the specified extensions
image_files = []
for ext in ('*.png', '*.jpg', '*.JPG', '*.PNG', '*.jpeg', '*.JPEG', '*.tif', '*.bmp', '*.BMP'):
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return image.point(table)

# Process each image file
for image_file in image_files:
    image = Image.open(image_file)

    # Split the image into R, G, B channels
    R, G, B = image.split()

    # Apply gamma correction and brightness enhancement
    IR = adjust_gamma(R, gamma)
    IG = adjust_gamma(G, gamma)
    IB = adjust_gamma(B, gamma)

    IR = ImageEnhance.Brightness(IR).enhance(a * b)
    IG = ImageEnhance.Brightness(IG).enhance(a * b)
    IB = ImageEnhance.Brightness(IB).enhance(a * b)

    # Merge the adjusted channels back into an image
    gamma_corrected_image = Image.merge("RGB", (IR, IG, IB))

    # Save the output image
    filename = os.path.basename(image_file)
    output_filename = os.path.join(output_folder, filename)
    gamma_corrected_image.save(output_filename)
