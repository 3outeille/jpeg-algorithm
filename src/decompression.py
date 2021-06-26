import itertools
import numpy as np
from numpy.lib.twodim_base import diag
import scipy as sp

from src.utils import Q_MAT, HUFFMAN_DC_TABLE_INV, HUFFMAN_AC_TABLE_INV, HUFFMAN_AC_TABLE
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

def huffman_inv(bitstream, largest_range):
    final_encoding = []

    beg, end = 0, 0
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

    return final_encoding

def entropy_coding_inv(bitstream, largest_range):
    # Huffman inverse
    final_encoding = huffman_inv(bitstream, largest_range)
    # Zigzag inverse
    q_block = zigzag_inv(final_encoding)
    return q_block

def bitstream_split(bitstream):
    EOB = HUFFMAN_AC_TABLE["0/0"]
    beg, ptr1, ptr2 = 0, 0, 0

    while ptr2 < len(bitstream):
        
        if ptr2 % 4 == 0:
            if bitstream[ptr1:ptr2] == EOB:
                yield bitstream[beg:ptr2]
                beg = ptr2
            ptr1 = ptr2
        
        ptr2 += 1


def decompress(bitstream, unpadding_values):

    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))
    
    # for bits_seq in bitstream_split(bitstream):
        # print(bits_seq)
        # ?? = entropy_coding(bits_seq, largest_range)
    
    print(next(bitstream_split(bitstream)))
    print(next(bitstream_split(bitstream)))

    # final_encoding = huffman_inv(bitstream, largest_range)

    # for chunk in range(0, len(final_encoding), 64):
    #     q_block = zigzag_inv(final_encoding[chunk: chunk + 64])
    #     print(q_block.shape)
    #     raise Exception("")

    # dct_block = quantization_inv(q_block, Q_MAT)
    # block = dct_inv(dct_block)
    # print(block)
    

# bitstream = "11000101"
# bitstream = ["11000101", "0100", "11100100" , "0101" ,"100001" ,"0110" , "100011", "001" ,"0100", "001", "001", "100101", "001" , "0110", "000" ,"001", "000", "0110", "11110100", "000", "1010"]
#             "11000101 | 100111010001011000010110100011101000100110010100101100000011101111010001010"
# bitstream = "11000101 | 0100 | 11100100 | 0101 | 100001 | 0110 | 100011 | 001 | 0100 | 001 | 001 | 100101 | 001 | 0110 | 000 | 001 | 000 | 0110 | 11110100 | 000 |  1010"
#              0      7   8  11  12     19 20  23 24    29 30  33 34    39 40 42 43  46 47 49 50 52 53    58 59 61 62  65 66 68 69 71 72 74 75  78 79      86 87  89 90  93     
#               -26        -3       0 -3     -2      -6      2       -4      1     -3     1     1      5       1      2     -1    1    -1     2     00000-1    -1      EOB
# decompress(bitstream)
