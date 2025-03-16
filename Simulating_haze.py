
from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import PIL.Image as pil
import cv2
import torch
import networks

def gen_haze(clean_img, depth_img, beta=1.0, A=150):
    depth_img_3c = np.stack([depth_img] * 3, axis=-1) / 255
    trans = np.exp(-depth_img_3c * beta)
    hazy = clean_img * trans + A * (1 - trans)
    return np.array(hazy, dtype=np.uint8)

def test_simple(image_path, model_name, ext, no_cuda, output_image_path, beta, airlight):
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

    model_path = os.path.join("models", model_name)
    encoder = networks.ResnetEncoder(18, False).to(device)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4)).to(device)

    encoder.load_state_dict(torch.load(os.path.join(model_path, "encoder.pth"), map_location=device))
    depth_decoder.load_state_dict(torch.load(os.path.join(model_path, "depth.pth"), map_location=device))
    encoder.eval()
    depth_decoder.eval()

    if os.path.isfile(image_path):
        paths = [image_path]
    elif os.path.isdir(image_path):
        paths = glob.glob(os.path.join(image_path, f'*.{ext}'))
    else:
        raise FileNotFoundError(f"Cannot find {image_path}")

    if not os.path.isdir(output_image_path):
        os.makedirs(output_image_path)

    with torch.no_grad():
        for image_path in paths:
            if image_path.endswith("_disp.jpg"):
                continue

            input_image = pil.open(image_path).convert('RGB')
            clean_img = np.array(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((640, 192), pil.LANCZOS)
            input_image = torch.tensor(np.array(input_image).transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(device)

            features = encoder(input_image)
            disp = depth_decoder(features)[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear",
                                                           align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            disp_resized_np = (disp_resized_np - disp_resized_np.min()) / (
                        disp_resized_np.max() - disp_resized_np.min()) * 255
            disp_resized_np = np.uint8(disp_resized_np)

            hazy = gen_haze(clean_img, disp_resized_np, beta=beta, A=airlight)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"{output_image_path}/{output_name}_hazy.jpg", cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
            print(f"Processed {image_path}")

    print(f'Done! Outputs saved in {output_image_path}')

# 修改这里以设置参数
image_path = "./data/test_images"  # 替换为你的图像路径或文件夹路径
model_name = "mono_640x192"  # 替换为你的预训练模型名称
ext = "png"  # 替换为你的图像扩展名
no_cuda = False  # 设置为 True 以禁用 CUDA
output_image_path = "./results"  # 替换为输出图像的文件夹路径
beta = 2.0  # 替换为你的雾霾程度
airlight = 200  # 替换为你的大气光强度

test_simple(image_path, model_name, ext, no_cuda, output_image_path, beta, airlight)
