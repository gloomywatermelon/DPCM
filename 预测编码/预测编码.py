import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def dpcm_encode(image):
    height, width = image.shape
    encoded_image = np.zeros((height, width), dtype=np.int16)
    predicted_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                predicted_image[i, j] = image[i, j]
                encoded_image[i, j] = image[i, j]
            elif i == 0:
                predicted_image[i, j] = image[i, j-1]
                encoded_image[i, j] = image[i, j] - image[i, j-1]
            elif j == 0:
                predicted_image[i, j] = image[i-1, j]
                encoded_image[i, j] = image[i, j] - image[i-1, j]
            else:
                predicted_image[i, j] = (image[i, j-1] + image[i-1, j]) // 2
                encoded_image[i, j] = image[i, j] - predicted_image[i, j]

    return encoded_image, predicted_image

def dpcm_decode(encoded_image, predicted_image):
    height, width = encoded_image.shape
    decoded_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                decoded_image[i, j] = predicted_image[i, j]
            elif i == 0:
                decoded_image[i, j] = encoded_image[i, j] + decoded_image[i, j-1]
            elif j == 0:
                decoded_image[i, j] = encoded_image[i, j] + decoded_image[i-1, j]
            else:
                decoded_image[i, j] = encoded_image[i, j] + (decoded_image[i, j-1] + decoded_image[i-1, j]) // 2

    return decoded_image

def quantize_image(image, step_size):
    quantized_image = np.round(image / step_size) * step_size
    return quantized_image

def main():
    # 读取灰度图像
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

    # 进行DPCM编码和解码
    encoded_image, predicted_image = dpcm_encode(image)
    decoded_image = dpcm_decode(encoded_image, predicted_image)

    # 比较不同量化器的重建图像区别并计算PSNR和SSIM值
    step_sizes = [1, 2, 4, 8]  # 不同的量化步长

    for step_size in step_sizes:
        quantized_image = quantize_image(encoded_image, step_size)
        decoded_quantized_image = dpcm_decode(quantized_image, predicted_image)

        psnr = peak_signal_noise_ratio(image, decoded_quantized_image)
        ssim = structural_similarity(image, decoded_quantized_image)

        print(f"量化步长: {step_size}")
        print(f"PSNR值: {psnr:.2f}")
        print(f"SSIM值: {ssim:.4f}")
        print()

        # 显示重建图像
        cv2.imshow(f"Reconstructed Image (Step Size: {step_size})", decoded_quantized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 保存重建图像
        cv2.imwrite(f"reconstructed_image_{step_size}.png", decoded_quantized_image)


if __name__ == '__main__':
    main()

                   

