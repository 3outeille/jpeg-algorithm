from src.compression import *
from src.decompression import *
from src.utils import rmse, snr

import time
import matplotlib.pyplot as plt

def main(img_path):
    
    # quality factor of 2quantization matrix.
    q = 50

    input_img = plt.imread(img_path)[..., :3] * 255
    print(input_img.shape)

    # plt.imshow(input_img / 255)
    # plt.show()

    t0 = time.time()
    bitstream, info_padding = compression(input_img, q=q, channel_mode="yuv")
    out_img = decompression(bitstream, info_padding, q=q, channel_mode="yuv")
    t1 = time.time()
    print(f"Compression + Decompression = {t1 - t0} s")

    # Plot
    plt.imshow(out_img / 255)
    plt.show()

    print(f"RMSE = {rmse(input_img, out_img)}")
    print(f"SNR = {snr(input_img, out_img)}")

# main("/home/sphird/Documents/codo-jpeg-algorithm/src/img/patrick-color.png")
# main("/home/sphird/Documents/codo-jpeg-algorithm/src/img/nyancat-patrick.png")
# main("/home/sphird/Documents/codo-jpeg-algorithm/src/img/pixil-frame.png")
# main("/home/sphird/Documents/codo-jpeg-algorithm/src/img/pikachu.png")
# main("/home/sphird/Documents/codo-jpeg-algorithm/src/img/gradient.png")