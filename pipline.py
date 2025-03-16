import os
import numpy as np
from PIL import Image, ImageEnhance
import torch
import networks
import cv2
import random
from torch.utils.data import Dataset

# Import necessary functions from the previous scripts
from generate_LQ import degrade  # Assuming degrade is available as mentioned in generate_LQ.py

# Darkness Simulation Functions
def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return image.point(table)

def simulate_darkness(image, gamma, a, b):
    R, G, B = image.split()
    IR = adjust_gamma(R, gamma)
    IG = adjust_gamma(G, gamma)
    IB = adjust_gamma(B, gamma)
    IR = ImageEnhance.Brightness(IR).enhance(a * b)
    IG = ImageEnhance.Brightness(IG).enhance(a * b)
    IB = ImageEnhance.Brightness(IB).enhance(a * b)
    darkened_image = Image.merge("RGB", (IR, IG, IB))
    return darkened_image

# Haze Simulation Functions
def gen_haze(clean_img, depth_img, beta=1.0, A=150):
    depth_img_3c = np.stack([depth_img] * 3, axis=-1) / 255
    trans = np.exp(-depth_img_3c * beta)
    hazy = clean_img * trans + A * (1 - trans)
    return np.array(hazy, dtype=np.uint8)

def simulate_haze(image, encoder, depth_decoder, device, beta=1.5, airlight=150):
    input_image = image.resize((1024, 320), Image.LANCZOS)
    input_image = torch.tensor(np.array(input_image).transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = encoder(input_image)
        disp = depth_decoder(features)[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, image.size[::-1], mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        disp_resized_np = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min()) * 255
        disp_resized_np = np.uint8(disp_resized_np)

    hazy = gen_haze(np.array(image), disp_resized_np, beta=beta, A=airlight)
    return Image.fromarray(hazy)

# Rain Simulation Functions
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
        rain_layer_resized = cv2.resize(rain_layer, (img.shape[1], img.shape[0]))
        img_with_rain = cv2.addWeighted(img.astype(np.float32) / 255.0, 0.7, rain_layer_resized, 0.4, 0) * 255.0
        return img_with_rain.astype(np.uint8)

# Function to load state dicts with ignored keys
def load_state_dict_ignore_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model.load_state_dict(filtered_state_dict, strict=False)

def process_image(input_image_path, output_image_path, encoder, depth_decoder, device, augmentor):
    # Load image
    image = Image.open(input_image_path).convert('RGB')

    # Randomly decide which steps to apply
    steps_to_apply = random.sample(['haze', 'degradation_noise', 'degradation_blur', 'degradation_jpeg', 'darkness', 'rain'], k=random.randint(1, 6))

    hazy_image = None
    degraded_image = None
    darkened_image = None
    final_image = None

    # Step 1: Simulate haze
    if 'haze' in steps_to_apply:
        hazy_image = simulate_haze(image, encoder, depth_decoder, device, beta=1.5, airlight=150)

    # Initialize the image to apply degradations on
    current_image = np.array(hazy_image) if hazy_image is not None else np.array(image)

    # Step 2: Apply degradations
    if 'degradation_noise' in steps_to_apply:
        current_image = degrade(current_image / 255.0, deg_type='noisy', param=25) * 255
    if 'degradation_blur' in steps_to_apply:
        current_image = degrade(current_image / 255.0, deg_type='blur', param=25) * 255
    if 'degradation_jpeg' in steps_to_apply:
        current_image = degrade(current_image / 255.0, deg_type='jpeg', param=25) * 255

    degraded_image = Image.fromarray(current_image.astype(np.uint8))

    # Step 3: Simulate darkness
    if 'darkness' in steps_to_apply:
        gamma = np.random.uniform(0.5, 2.0)
        a = np.random.uniform(0.5, 1)
        b = np.random.uniform(0.5, 1)
        darkened_image = simulate_darkness(degraded_image, gamma, a, b)
    else:
        darkened_image = degraded_image

    # Step 4: Simulate rain
    if 'rain' in steps_to_apply:
        final_image_cv2 = np.array(darkened_image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        rand_id1 = random.randint(1, 165)
        rand_id2 = random.randint(4, 8)
        rain_augmented_image = augmentor.apply_rain_streaks(final_image_cv2, rand_id1, rand_id2)
        final_image = Image.fromarray(rain_augmented_image[:, :, ::-1])  # Convert BGR back to RGB
    else:
        final_image = darkened_image

    # Save the final image
    final_image.save(output_image_path)


def process_images(input_folder, output_folder, model_name="mono+stereo_1024x320", no_cuda=False):
    # Setup device and load models
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model_path = os.path.join("models", model_name)
    encoder = networks.ResnetEncoder(18, False).to(device)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4)).to(device)

    # Load state dicts
    encoder_state_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    depth_decoder_state_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)

    # Load state dicts with ignored keys
    load_state_dict_ignore_mismatch(encoder, encoder_state_dict)
    load_state_dict_ignore_mismatch(depth_decoder, depth_decoder_state_dict)

    encoder.eval()
    depth_decoder.eval()

    # Initialize the RainStreakAugmentor
    augmentor = RainStreakAugmentor(root_dir="Streaks_Garg06")

    # Process all images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            process_image(input_image_path, output_image_path, encoder, depth_decoder, device, augmentor)
            print(f"Processed and saved: {filename}")

# Example usage
if __name__ == '__main__':
    input_folder = 'FOD-A'
    output_folder = 'FOD-B'
    process_images(input_folder, output_folder)
