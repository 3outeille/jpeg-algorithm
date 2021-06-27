import itertools
import numpy as np
import scipy as sp
import scipy.fft

from src.utils import Q_MAT, Q_MAT_UV, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE, LARGEST_RANGE
from src.utils import decimal_to_binary

def padding(img, info_padding, mode="replicate"):
    """
        Returns padded image where image is centered in middle.

        @Params:
        - img: input image of shape (n, m).
        - info_padding: dictionary with padding informations.
        - mode: padding mode.
    """
    # Round to upper-nearest multiple of 8.
    w = (img.shape[0] + 7) & (-8)
    h = (img.shape[1] + 7) & (-8)

    # Compute padding for image to be centered in middle.
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

    # Padding
    if mode == "black":
        padded_img = np.pad(img, [(ax1_top , ax1_bot), (ax2_left, ax2_right)], 'constant')
    elif mode == "replicate":
        padded_img = np.pad(img, [(ax1_top, ax1_bot), (ax2_left, ax2_right)], 'symmetric')
    else:
        raise ValueError("This mode doesn't exist")

    # Fill info_padding dictionary.
    for key, val in zip(info_padding.keys(), [ax1_top , ax1_bot, ax2_left, ax2_right]):
        info_padding[key].append(val)

    info_padding["img_padded_shape"] = padded_img.shape

    return padded_img

def block_splitting(img):
    """
        Yield 8x8 macroblock from image.

        @Params:
        - img: input image of shape (n, m).
    """
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            yield img[i:i+8, j:j+8]

def dct(block):
    """
        Returns macroblock with DCT applied to it.

        @Params:
        - block: 8x8 macroblock.
    """
    block =  block - 128
    dct_block = sp.fft.dct(block, axis=0, type=2, norm="ortho")
    dct_block = sp.fft.dct(dct_block, axis=1, type=2, norm="ortho")
    return dct_block

def quantization(dct_block, Q_MAT, q=50):
    """
        Returns quantized macroblock.

        @Params:
        - dct_block: 8x8 macroblock with DCT applied to it.
        - Q_MAT: Hardcoded quantization matrix.
        - q: quality factor of quantization matrix in range of [1, 100].
    """
    if q < 1 or q > 100:
        raise ValueError("Invalid q value. Should be in range of [1, 100]")
    
    # Compression factor quality.
    if q < 50:
        a = 5000/q
    else:
        a = 200 - 2*q
    
    Qq = np.floor((a*Q_MAT + 50) / 100)
    q_block = np.divide(dct_block, Qq)
    q_block = np.rint(q_block).astype(int)
    return q_block

def zigzag(q_block):
    """
        Returns zigzag ordering of q_block.

        @Params:
        - q_block: quantized 8x8 macroblock.
    """
    n, m = q_block.shape
    res = [[] for i in range(n + m - 1)]

    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0:
                res[i + j].insert(0, q_block[i][j])
            else:
                res[i + j].append(q_block[i][j])

    return np.trim_zeros(np.concatenate(res), trim='b')


def rgb2yuv(img):
    c, n, m = img.shape
    Y = np.zeros((n, m))
    U = np.zeros((n, m))
    V = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            Y[i][j] =  0.299 * img[0][i][j] + 0.587 * img[1][i][j] + 0.114 * img[2][i][j]
            U[i][j] = -0.14713 * img[0][i][j] - 0.28886 * img[1][i][j] + 0.436 * img[2][i][j]
            V[i][j] =  0.615 * img[0][i][j] - 0.51499 * img[1][i][j] - 0.10001 * img[2][i][j]
            U[i][j] = max(U[i][j], 0.436) if U[i][j] > 0 else min(U[i][j], -0.436)
            V[i][j] = max(U[i][j], 0.615) if U[i][j] > 0 else min(U[i][j], -0.615)

    img[0] = Y
    img[1] = U
    img[2] = V
    return img

def huffman(zigzag_order, LARGEST_RANGE):
    """
        Returns encoded macroblock from zigzag ordering.

        @Params:
        - zigzag_order: zigzag ordering of quantized block.
        - LARGEST_RANGE: JPEG coefficient coding categories. Only category range 15.
    """
    final_encoding = []

    # DC coeff encoding
    dc_coeff = zigzag_order[0]
    CAT, binary = decimal_to_binary(dc_coeff, LARGEST_RANGE)
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
        
        CAT, binary = decimal_to_binary(ac_coeff, LARGEST_RANGE)
        codeword = HUFFMAN_AC_TABLE[f"{RUN}/{CAT}"]
        final_encoding.append(codeword + binary)
        RUN = 0
    
    # Add end of block.
    EOB = "0/0"
    final_encoding.append(HUFFMAN_AC_TABLE[EOB])
    return final_encoding

def compression(img, q=50, mode="replicate", channel_mode="rgb"):
    """
        Returns bitstream representing compressed image.

        @Params:
        - img: input image of shape (n, m, c).
        - q: quality factor of quantization matrix in range of [1, 100].
        - mode: padding mode.
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image dimension is not supported")
    
    img = np.transpose(img, (2, 0, 1))
    
    # Padding information needed during decompression (mutable object).
    info_padding = {
        "ax1_top": [],
        "ax1_bot": [],
        "ax2_left": [], 
        "ax2_right": [],
        "img_padded_shape": None
    }

    bitstream = []

    if channel_mode == "yuv":
        img = rgb2yuv(img)

    for channel in range(3):
        # Padd each image channel.
        img_channel = padding(img[channel, ...], info_padding, mode)

        # Step 1: Block splitting
        for block in block_splitting(img_channel):
            # Step 2: Discrete cosine transform (DCT)
            dct_block = dct(block)
            # Step 3: Quantization + Round to nearest integer
            mat = Q_MAT_UV if (channel != 0 and channel_mode == "yuv") else Q_MAT
            q_block = quantization(dct_block, mat, q)
            # Step 4: Zigzag + Huffman
            zigzag_order = zigzag(q_block)
            final_encoding = huffman(zigzag_order, LARGEST_RANGE)
            
            bitstream.append(final_encoding)

    out = "".join(map(str, np.concatenate(bitstream)))
    return out, info_padding
