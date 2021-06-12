import itertools

from src.utils import HUFFMAN_DC_TABLE_INV
from src.utils import binary_to_decimal

def dct_inv():
    pass

def quantization_inv():
    pass

def zigzag_inv():
    pass

def huffman_inv(bitstream, largest_range):
    res = []

    # Retrieve DC coeff
    beg, end = 0, 0
    while end < len(bitstream):
        if bitstream[beg:end] in HUFFMAN_DC_TABLE_INV:
            CAT = HUFFMAN_DC_TABLE_INV[bitstream[beg:end]]
            break
        end += 1

    dc_coeff = binary_to_decimal(bitstream[end: end + CAT], largest_range)
    res.append(dc_coeff)

    # Retrieve AC coeff

    return res

def decompress(bistream):

    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))

    res = huffman_inv(bistream, largest_range)
    print(res)

bitstream_26 = "11000101"
# bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"
decompress(bitstream_26)