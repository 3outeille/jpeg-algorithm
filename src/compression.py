import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from src.utils import Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE
from src.utils import retrieve_binary_rep, save_img

def padding(img, mode="black"):
    """
        Only work for 2d matrix.
    """
    # Round up to nearest multiple of 8
    w = (img.shape[0] + 7) & (-8)
    h = (img.shape[1] + 7) & (-8)

    # Smart padding -> Pad with image center in middle.
    delta_w = w - img.shape[0]
    delta_h = h - img.shape[1]
    
    ax1_top, ax1_bot, ax2_left, ax2_right = 0, 0, 0, 0

    if delta_w != 0:
        ax1_top, ax1_bot = (delta_w//2) , (delta_w//2)
        if delta_w % 2 == 1:    
            ax1_bot = (delta_w//2) + 1
    
    if delta_h != 0:
        ax2_left, ax2_right = (delta_h//2) , (delta_h//2)
        if delta_h % 2 == 1:
            ax2_right = (delta_h//2) + 1

    if mode == "black":
        img = np.pad(img, [(ax1_top , ax1_bot), (ax2_left, ax2_right)], 'constant')
    elif mode == "replicate":
        img = np.pad(img, [(ax1_top, ax1_bot), (ax2_left, ax2_right)], 'symmetric')
    else:
        raise ValueError("This mode doesn't exist")

    return img

def block_splitting(img):
    """
        8x8 sliding window over image

        TODO: yield macro block later ?
    """
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            yield img[i:i+8, j:j+8]

def dct(block):
    block =  block - 128
    dct_block = sp.fft.dct(block, axis=0, type=2, norm="ortho")
    dct_block = sp.fft.dct(dct_block, axis=1, type=2, norm="ortho")
    return dct_block

def quantization(dct_block, Q):
    return np.rint(np.divide(dct_block, Q)).astype(int)

def zigzag(q_block):
    n, m = q_block.shape
    res = [[] for i in range(n + m - 1)]
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0:
                res[i + j].insert(0, q_block[i][j])
            else:
                res[i + j].append(q_block[i][j])

    return np.trim_zeros(np.concatenate(res), trim='b')

def huffman(largest_range, zigzag_order):
    final_encoding = []

    # DC coeff encoding
    dc_coeff = zigzag_order[0]
    CAT, binary = retrieve_binary_rep(largest_range, dc_coeff)
    codeword = HUFFMAN_DC_TABLE[CAT]
    final_encoding.append(codeword + binary)

    # AC coeff encoding: Run Length Encoding
    RUN = 0
    for ac_coeff in zigzag_order[1:]:
        if ac_coeff == 0:
            RUN += 1
            continue
        CAT, binary = retrieve_binary_rep(largest_range, ac_coeff)
        codeword = HUFFMAN_AC_TABLE[f"{RUN}/{CAT}"]
        final_encoding.append(codeword + binary)
        RUN = 0
    
    # Add end of block.
    EOB = "0/0"
    final_encoding.append(HUFFMAN_AC_TABLE[EOB])
    return final_encoding

def entropy_coding(largest_range, q_block):
    # Zigzag
    zigzag_order = zigzag(q_block)
    # Huffman
    final_encoding = huffman(largest_range, zigzag_order)
    return final_encoding


def compression(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image dimension is not supported")
    
    # JPEG coefficient coding category 15
    largest_range = list(itertools.product(['0', '1'], repeat=15))

    img = np.transpose(img, (2, 0, 1))
    
    bitstream = []
    for channel in range(3):
        # Step 1: Block splitting
        img_channel = padding(img[channel, ...], mode="replicate")

        for block in block_splitting(img_channel):
            # Step 2: Discrete cosine transform (DCT)
            dct_block = dct(block)
            # Step 3: Quantization + Round to nearest integer
            q_block = quantization(dct_block, Q_MAT)
            # Step 4: Zigzag + Huffman
            final_encoding = entropy_coding(largest_range, q_block)
            bitstream.append(final_encoding)

    return "".join(map(str, np.concatenate(bitstream)))

img = plt.imread("nyancat-patrick.png")
bitstream = compression(img)
save_img(bitstream, "compressed-nyancat.jpg")