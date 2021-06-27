import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from src.utils import Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE, load_img
from src.utils import decimal_to_binary, save_img

def padding(img, unpadding_values, mode="black"):
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

    for key, val in zip(unpadding_values.keys(), [ax1_top , ax1_bot, ax2_left, ax2_right]):
        unpadding_values[key].append(val)

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
    q_block = np.divide(dct_block, Q)
    return np.rint(q_block).astype(int)

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

def huffman(zigzag_order, largest_range):
    final_encoding = []

    # DC coeff encoding
    dc_coeff = zigzag_order[0]
    CAT, binary = decimal_to_binary(dc_coeff, largest_range)
    codeword = HUFFMAN_DC_TABLE[CAT]
    final_encoding.append(codeword + binary)

    # AC coeff encoding: Run Length Encoding
    RUN = 0
    for ac_coeff in zigzag_order[1:]:
        if ac_coeff == 0:
            RUN += 1
            continue
        
        while RUN - 15 > 0:
            # Split in batch of "15/0".
            codeword = HUFFMAN_AC_TABLE["15/0"]
            final_encoding.append(codeword + "0")
            RUN -= 15
        
        CAT, binary = decimal_to_binary(ac_coeff, largest_range)
        codeword = HUFFMAN_AC_TABLE[f"{RUN}/{CAT}"]
        final_encoding.append(codeword + binary)
        RUN = 0
    
    # Add end of block.
    EOB = "0/0"
    final_encoding.append(HUFFMAN_AC_TABLE[EOB])
    return final_encoding

def compression(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image dimension is not supported")
    
    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))

    img = np.transpose(img, (2, 0, 1))
    
    # Need it during decompression (Mutable object)
    unpadding_values = {
        "ax1_top": [],
        "ax1_bot": [],
        "ax2_left": [], 
        "ax2_right": []
    }

    bitstream = []
    for channel in range(3):
        img_channel = padding(img[channel, ...], unpadding_values, mode="replicate")

        # Step 1: Block splitting
        for block in block_splitting(img_channel):
            # Step 2: Discrete cosine transform (DCT)
            dct_block = dct(block)
            # Step 3: Quantization + Round to nearest integer
            q_block = quantization(dct_block, Q_MAT)
            # Step 4: Zigzag + Huffman
            zigzag_order = zigzag(q_block)
            final_encoding = huffman(zigzag_order, largest_range)
            
            bitstream.append(final_encoding)

    out = "".join(map(str, np.concatenate(bitstream)))
    return out, unpadding_values

# import matplotlib.pyplot as plt
# img = plt.imread("nyancat-patrick.png")
# bitstream, unpaddi_values = compression(img)
# print(len(bitstream))
# save_img(bitstream, "patrick-compresed.jpg")
# bitstream = ["11000101", "0100", "11100100" , "0101" ,"100001" ,"0110" , "100011", "001" ,"0100", "001", "001", "100101", "001" , "0110", "000" ,"001", "000", "0110", "11110100", "000", "1010"]
# bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"