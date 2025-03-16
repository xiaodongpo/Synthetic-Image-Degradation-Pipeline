import numpy as np
import cv2
import os
import random
import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#blur
def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((
        xx.reshape((kernel_size * kernel_size, 1)),
        yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         noise_range=None):
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range,
            noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range,
            noise_range=noise_range, isotropic=False)
    return kernel

#noise
def generate_gaussian_noise(img, sigma=10, gray_noise=False):
    if gray_noise:
        noise = np.float32(np.random.randn(*(img.shape[0:2]))) * sigma / 255.
        noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    return noise


def add_gaussian_noise(img, sigma=10, clip=True, rounds=False, gray_noise=False):
    noise = generate_gaussian_noise(img, sigma, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

#jpeg
def add_jpg_compression(img, quality=90):
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img

#degrade
def degrade(img, deg_type, param=15):
    if deg_type == 'noisy':
        output = add_gaussian_noise(img, sigma=param)
    elif deg_type == 'blur':
        kernel = random_mixed_kernels(['iso'], [1], kernel_size=param)
        output = cv2.filter2D(img, -1, kernel)
    elif deg_type == 'jpeg':
        output = add_jpg_compression(img, param)

    return output



IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# deg_type: noisy, jpeg
# param: 50 for noise_level, 10 for jpeg compression quality
def generate_LQ(deg_type='blur', param=50):
    print(deg_type, param)
    # set data dir
    sourcedir = "datasets/universal/val/noisy/GT"
    savedir = "datasets/universal/val/noisy/LQ"

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    filepaths = [f for f in os.listdir(sourcedir) if is_image_file(f)]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename)) / 255.

        image_LQ = (degrade(image, deg_type, param) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(savedir, filename), image_LQ)
        
    print('Finished!!!')

if __name__ == "__main__":
    generate_LQ()
