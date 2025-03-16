import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset


class RainStreakAugmentor(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def getRainLayer(self, rand_id1, rand_id2):
        """Load a rain streak layer given two random IDs."""
        path_img_rainlayer_src = os.path.join(self.root_dir, f"{rand_id1}-{rand_id2}.png")

        rainlayer_rand = cv2.imread(path_img_rainlayer_src)

        rainlayer_rand = rainlayer_rand.astype(np.float32) / 255.0
        rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
        return rainlayer_rand

    def apply_rain_streaks(self, img, rand_id1, rand_id2):
        """Apply a rain streak layer to the input image."""

        rain_layer = self.getRainLayer(rand_id1, rand_id2)

        # Resize rain layer to match the input image size
        rain_layer_resized = cv2.resize(rain_layer, (img.shape[1], img.shape[0]))

        # Blend the images
        img_with_rain = cv2.addWeighted(img.astype(np.float32) / 255.0, 0.7, rain_layer_resized, 0.4, 0) * 255.0
        return img_with_rain.astype(np.uint8)


# Example usage
if __name__ == "__main__":
    # Define the root directory where rain streak images are stored
    root_dir = "Streaks_Garg06"

    # Initialize the RainStreakAugmentor
    augmentor = RainStreakAugmentor(root_dir=root_dir)

    # Define the directory containing the images (gt2 folder)
    img_dir = "gt2"
    output_dir = "lq2"  # Directory to save the augmented images

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each image file in the directory
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        # Ensure the file is an image (basic check based on extension)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load the image
            img_clean = cv2.imread(img_path)
            if img_clean is None:
                print(f"Warning: The image '{img_path}' could not be loaded. Skipping...")
                continue

            # Apply rain streaks with random IDs
            rand_id1 = random.randint(1, 165)
            rand_id2 = random.randint(4, 8)
            img_with_rain = augmentor.apply_rain_streaks(img_clean, rand_id1, rand_id2)

            # Save the augmented image
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, img_with_rain)
            print(f"Saved augmented image to '{output_path}'")
