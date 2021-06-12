import itertools
import numpy as np
import scipy as sp

from src.utils import Q_MAT, HUFFMAN_DC_TABLE_INV, HUFFMAN_AC_TABLE_INV
from src.utils import binary_to_decimal

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
    nb_square, diagonal, ptr, n = 0, 0, 0, len(final_encoding)
    
    while nb_square < n:
        idx_i = [i for i in range(diagonal + 1)]
        idx_j = [j for j in range(diagonal, -1, -1)]

        for i, j in zip(idx_i, idx_j):
            if ptr < n:
                if diagonal % 2 == 1:
                    q_block[i][j] = final_encoding[ptr]
                else: # flip
                    q_block[j][i] = final_encoding[ptr]
                ptr += 1
            else:
                break

        nb_square += diagonal + 1
        diagonal += 1

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

    while end < len(bitstream):
        # Retrieve AC coeff
        while end < len(bitstream) + 1:
            codeword = bitstream[beg:end]
            if codeword in HUFFMAN_AC_TABLE_INV:
                RUN_CAT = HUFFMAN_AC_TABLE_INV[codeword]
                break
            end += 1

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

def decompress(bistream):

    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))
    
    q_block = entropy_coding_inv(bitstream, largest_range)
    dct_block = quantization_inv(q_block, Q_MAT)
    block = dct_inv(dct_block)
    print(block)
    

# bitstream = "11000101"
# bitstream = ["11000101", "0100", "11100100" , "0101" ,"100001" ,"0110" , "100011", "001" ,"0100", "001", "001", "100101", "001" , "0110", "000" ,"001", "000", "0110", "11110100", "000", "1010"]
#             "11000101 | 100111010001011000010110100011101000100110010100101100000011101111010001010"
# bitstream = "11000101 | 0100 | 11100100 | 0101 | 100001 | 0110 | 100011 | 001 | 0100 | 001 | 001 | 100101 | 001 | 0110 | 000 | 001 | 000 | 0110 | 11110100 | 000 |  1010"
#              0      7   8  11  12     19 20  23 24    29 30  33 34    39 40 42 43  46 47 49 50 52 53    58 59 61 62  65 66 68 69 71 72 74 75  78 79      86 87  89 90  93     
#               -26        -3       0 -3     -2      -6      2       -4      1     -3     1     1      5       1      2     -1    1    -1     2     00000-1    -1      EOB
# decompress(bitstream)