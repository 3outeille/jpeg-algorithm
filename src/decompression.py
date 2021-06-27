import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.records import array
import scipy as sp

from src.utils import Q_MAT, HUFFMAN_DC_TABLE_INV, HUFFMAN_AC_TABLE_INV
from src.utils import binary_to_decimal

def block_combination():
    pass

def unpadding(img, unpadding_values):
    # n, m = img.shape
    # # og_img = ?

    # for channel in range(3):
    #     ax1_top = unpadding_values["ax1_top"][channel]
    #     ax1_bot = unpadding_values["ax1_bot"][channel]
    #     ax2_left = unpadding_values["ax2_left"][channel]
    #     ax2_right = unpadding_values["ax2_right"][channel]
        
    #     og_img = img[ax1_top:n-ax1_bot, ax2_left:m-ax2_right]

    # return og_img
    pass


def dct_inv(dct_block):
    block = sp.fft.dct(dct_block, axis=0, type=3, norm="ortho")
    block = sp.fft.dct(block, axis=1, type=3, norm="ortho")
    block = block + 128
    block = np.rint(block).astype(int)
    block = np.clip(block, 0, 255)
    return block

def quantization_inv(q_block, Q_MAT):
    dct_block = np.multiply(q_block, Q_MAT)
    return dct_block

def zigzag_inv(final_encoding):
    q_block = np.zeros((8, 8), dtype=int)
    diagonal, ptr, n = 0, 0, len(final_encoding)
    a, b = 0, -1

    while ptr < n:
      
        idx_i = [i for i in range(a, diagonal + 1)]
        idx_j = [j for j in range(diagonal, b, -1)]

        for i, j in zip(idx_i, idx_j):
            if ptr < n:
                if (diagonal - a) % 2 == 1:
                    q_block[i][j] = final_encoding[ptr]
                else: # flip
                    q_block[j][i] = final_encoding[ptr]
                ptr += 1
            else:
                break

        if diagonal < 7:
            diagonal += 1
        else:
            a += 1
            b += 1

    return q_block

def huffman_inv(bitstream, largest_range, beg=0, end=0):

    while end < len(bitstream):
        final_encoding, beg, end = huffman_inv_aux(bitstream, largest_range, beg, end)
        yield final_encoding

def huffman_inv_aux(bitstream, largest_range, beg, end):
    final_encoding = []

    # Retrieve DC coeff
    while end < len(bitstream):
        codeword = bitstream[beg:end]
        if codeword in HUFFMAN_DC_TABLE_INV:
            CAT = HUFFMAN_DC_TABLE_INV[codeword]
            break
        end += 1

    dc_coeff = binary_to_decimal(bitstream[end: end + CAT], largest_range)
    final_encoding.append(dc_coeff)
    
    end += CAT
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

        ac_coeff = binary_to_decimal(bitstream[end: end + CAT], largest_range)

        for i in range(RUN):
            final_encoding.append(0)
            
        final_encoding.append(ac_coeff)

        end += CAT
        beg = end

    return final_encoding, beg, end

def decompress(bitstream, unpadding_values):

    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))
    
    result, channel = [], []

    for i, final_encoding in enumerate(huffman_inv(bitstream, largest_range)):
        q_block = zigzag_inv(final_encoding)
        dct_block = quantization_inv(q_block, Q_MAT)
        block = dct_inv(dct_block)
        channel.append(block)

        if (i + 1) % 4 == 0:
            result.append(channel)
            channel = []
    
    result = np.array(result)
    channel1 = result[:4, ...]
    channel2 = result[4:8, ...]
    channel3 = result[8:12, ...]

    print(channel1.shape)
    print(channel2.shape)
    print(channel3.shape)

    channel1 = np.concatenate(channel1, axis=1)
    channel1 = channel1.transpose(0, 2, 1)
    channel1 = np.concatenate(channel1, axis=0)
    plt.imshow(channel1)
    plt.show()

    channel2 = np.concatenate(channel2, axis=1)
    channel2 = channel2.transpose(0, 2, 1)
    channel2 = np.concatenate(channel2, axis=0)
    plt.imshow(channel2)
    plt.show()

    channel3 = np.concatenate(channel3, axis=1)
    channel3 = channel3.transpose(0, 2, 1)
    channel3 = np.concatenate(channel3, axis=0)
    plt.imshow(channel3)
    plt.show()


# bitstream = "11000101"
# bitstream = ["11000101", "0100", "11100100" , "0101" ,"100001" ,"0110" , "100011", "001" ,"0100", "001", "001", "100101", "001" , "0110", "000" ,"001", "000", "0110", "11110100", "000", "1010"]
#             "11000101 | 100111010001011000010110100011101000100110010100101100000011101111010001010"
# bitstream = "11000101 | 0100 | 11100100 | 0101 | 100001 | 0110 | 100011 | 001 | 0100 | 001 | 001 | 100101 | 001 | 0110 | 000 | 001 | 000 | 0110 | 11110100 | 000 |  1010"
#              0      7   8  11  12     19 20  23 24    29 30  33 34    39 40 42 43  46 47 49 50 52 53    58 59 61 62  65 66 68 69 71 72 74 75  78 79      86 87  89 90  93     
#               -26        -3       0 -3     -2      -6      2       -4      1     -3     1     1      5       1      2     -1    1    -1     2     00000-1    -1      EOB
# decompress(bitstream)
