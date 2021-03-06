import numpy as np
import scipy as sp
from skimage.color import yuv2rgb

from src.utils import Q_MAT, Q_MAT_UV, HUFFMAN_DC_TABLE_INV, HUFFMAN_AC_TABLE_INV, LARGEST_RANGE
from src.utils import binary_to_decimal

def huffman_inv(bitstream, LARGEST_RANGE, beg=0, end=0):
    """
        Yield macroblock in zigzag order from input bistream.

        @Params:
        - bitstream: string representing compressed image.
        - LARGEST_RANGE: JPEG coefficient coding categories. Only category range 15.
    """
    while end < len(bitstream):
        final_encoding, beg, end = huffman_inv_aux(bitstream, LARGEST_RANGE, beg, end)
        yield final_encoding   

def huffman_inv_aux(bitstream, LARGEST_RANGE, beg, end):
    """
        Returns macroblock in zigzag order from input bistream.
        @Params:
        - bitstream: string representing compressed image.
        - LARGEST_RANGE: JPEG coefficient coding categories. Only category range 15.
    """
    zigzag_order = []

    # Retrieve DC coeff
    while end < len(bitstream):
        codeword = bitstream[beg:end]
        if codeword in HUFFMAN_DC_TABLE_INV:
            CAT = HUFFMAN_DC_TABLE_INV[codeword]
            break
        end += 1

    dc_coeff = binary_to_decimal(bitstream[end: end + CAT], LARGEST_RANGE)
    zigzag_order.append(dc_coeff)
    
    end += max(CAT, 1)
    beg = end

    EOB = "1010"

    while end < len(bitstream):
        # Retrieve AC coeff
        codeword = bitstream[beg:end]
        while (codeword not in HUFFMAN_AC_TABLE_INV):
            end += 1
            codeword = bitstream[beg:end]

        if codeword == EOB:
            beg = end
            break

        RUN_CAT = HUFFMAN_AC_TABLE_INV[codeword]
        RUN, CAT = RUN_CAT.split("/")
        RUN, CAT = int(RUN), int(CAT)

        ac_coeff = binary_to_decimal(bitstream[end: end + CAT], LARGEST_RANGE)

        for i in range(RUN):
            zigzag_order.append(0)
            
        zigzag_order.append(ac_coeff)

        end += max(CAT, 1)
        beg = end

    return zigzag_order, beg, end

def zigzag_inv(zigzag_order):
    """
        Returns quantized block from zigzag ordered block.
        @Params:
        - zigzag_order: zigzag ordered block.
    """
    q_block = np.zeros((8, 8), dtype=int)
    diagonal, ptr, n = 0, 0, len(zigzag_order)
    a, b = 0, -1

    while ptr < n:
      
        idx_i = [i for i in range(a, diagonal + 1)]
        idx_j = [j for j in range(diagonal, b, -1)]

        for i, j in zip(idx_i, idx_j):
            if ptr < n:
                if (diagonal - a) % 2 == 1:
                    q_block[i][j] = zigzag_order[ptr]
                else: # flip
                    q_block[j][i] = zigzag_order[ptr]
                ptr += 1
            else:
                break

        if diagonal < 7:
            diagonal += 1
        else:
            a += 1
            b += 1

    return q_block

def quantization_inv(q_block, mat, q=50):
    """
        Returns unquantized macroblock.

        @Params:
        - q_block: 8x8 quantized macroblock.
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

    Qq = np.floor((a*mat + 50) / 100)
    dct_block = np.multiply(q_block, Qq)
    return dct_block

def dct_inv(dct_block):
    """
        Returns macroblock with DCT not applied to it.

        @Params:
        - dct_block: 8x8 macroblock with DCT applied to it.
    """

    block = sp.fft.dct(dct_block, axis=0, type=3, norm="ortho")
    block = sp.fft.dct(block, axis=1, type=3, norm="ortho")
    block = block + 128
    block = np.rint(block).astype(int)
    block = np.clip(block, 0, 255)
    return block

def unpadding(img, info_padding):
    """
        Returns unpadded image.

        @Params:
        - img: input image of shape (n, m).
        - info_padding: dictionary with padding informations.
    """
    n, m, channels = img.shape

    for channel in range(channels):
        ax1_top = info_padding["ax1_top"][channel]
        ax1_bot = info_padding["ax1_bot"][channel]
        ax2_left = info_padding["ax2_left"][channel]
        ax2_right = info_padding["ax2_right"][channel]
        
        # Remove padding.
        og_img = img[ax1_top:n-ax1_bot, ax2_left:m-ax2_right]

    return og_img

def upsampling(img, downsampling_mode):
    """
        Upsampling the channel U and V

        @Params:
        - img: image to upsample
        - mode: mode of upsampling (4:4:4, 4:2:2, 4:2:0)
    """
    if downsampling_mode == "4:4:4":
        return

    for i in range(img.shape[1] // 2 - 1, 0, -1):
        for j in range(img.shape[0]):
            img[...,1][j][2 * i] = img[...,1][j][i]
            img[...,1][j][(2 * i) + 1] = img[...,1][j][i]
            img[...,2][j][2 * i] = img[...,2][j][i]
            img[...,2][j][(2 * i) + 1] = img[...,2][j][i]

    if downsampling_mode == "4:2:0":
        for i in range(img.shape[0] // 2 - 1, 0, -1):
            img[...,1][2 * i] = img[...,1][i]
            img[...,1][(2 * i) + 1] = img[...,1][i]
            img[...,2][2 * i] = img[...,2][i]
            img[...,2][(2 * i) + 1] = img[...,2][i]


def decompression(bitstream, info_padding, q=50, channel_mode="rgb", downsampling_mode="4:4:4"):
    """
        Returns decompressed image from bitstream.
        
        @Params:
        - bitstream: string representing compressed image.
        - info_padding: dictionary with padding informations.
        - q: quality factor of quantization matrix.
    """
    n, m, c = *info_padding["img_padded_shape"], 3
    frame = np.zeros((n, m, c))

    nb_max_block_per_row = n // 8
    nb_max_block_per_col = m // 8
    nb_total_block_per_channel = nb_max_block_per_row * nb_max_block_per_col
    row, col, channel = 0, 0, 0

    # Step 1: Huffman inverse.
    for nb_block, zigzag_order in enumerate(huffman_inv(bitstream, LARGEST_RANGE)):

        # Step 1: Zigzag inverse.
        q_block = zigzag_inv(zigzag_order)
        mat = Q_MAT_UV if (channel != 0 and channel_mode == "yuv") else Q_MAT
        # Step 2: Quantization inverse.
        dct_block = quantization_inv(q_block, mat, q)
        # Step 3: DCT inverse.
        block = dct_inv(dct_block)
        # Step 4: Block combination
        frame[row:row+8, col:col+8, channel] = block

        col += 8

        # Go to next row.
        if ((nb_block + 1) % nb_max_block_per_col) == 0:
            row += 8
            col = 0

        # Go to next channel.
        if ((nb_block + 1) % nb_total_block_per_channel) == 0:
            channel += 1
            row, col = 0, 0

    upsampling(frame, downsampling_mode)

    # Unpadding.
    img = unpadding(frame, info_padding)

    if channel_mode == "yuv":
        img = yuv2rgb(img)

    return img
