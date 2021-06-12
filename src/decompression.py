import itertools

from src.utils import HUFFMAN_DC_TABLE_INV, HUFFMAN_AC_TABLE_INV
from src.utils import binary_to_decimal

def dct_inv():
    pass

def quantization_inv():
    pass

def zigzag_inv():
    pass

def huffman_inv(bitstream, largest_range):
    res = []

    beg, end = 0, 0
    # Retrieve DC coeff
    while end < len(bitstream):
        codeword = bitstream[beg:end]
        if codeword in HUFFMAN_DC_TABLE_INV:
            CAT = HUFFMAN_DC_TABLE_INV[codeword]
            break
        end += 1

    dc_coeff = binary_to_decimal(bitstream[end: end + CAT], largest_range)
    res.append(dc_coeff)
    end += CAT
    beg = end

    while end < len(bitstream):
        # Retrieve AC coeff
        while end < len(bitstream) + 1:
            codeword = bitstream[beg:end]
            if codeword in HUFFMAN_AC_TABLE_INV:
                print(f"codeword = {codeword}")
                RUN_CAT = HUFFMAN_AC_TABLE_INV[codeword]
                break
            end += 1

        RUN, CAT = RUN_CAT.split("/")
        RUN, CAT = int(RUN), int(CAT)

        print(f"beg = {beg} | end = {end}")
        ac_coeff = binary_to_decimal(bitstream[end: end + CAT], largest_range)
        print(f"ac_coeff: {ac_coeff} | RUN/CAT {RUN_CAT}")

        for i in range(RUN):
            res.append(0)
            
        res.append(ac_coeff)

        end += CAT
        print(beg, end, len(bitstream))
        print(res)
        print("-----")
        beg = end

    return res

def decompress(bistream):

    # JPEG coefficient coding category 15
    # FIXME: Maybe precomputed it like Q_MAT, HUFFMAN_DC_TABLE, HUFFMAN_AC_TABLE ?
    largest_range = list(itertools.product(['0', '1'], repeat=15))

    res = huffman_inv(bistream, largest_range)
    print(res)

# bitstream = "11000101"
bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"
# bitstream = "11000101 | 0100 | 11100100 | 0101 | 100001 | 0110 | 100011 | 001 | 0100 | 001 | 001 | 100101 | 001 | 0110 | 000 | 001 | 000 | 0110 | 11110100 | 000 |  1010"
#              0      7   8  11  12     19 20  23 24    29 30  33 34    39 40 42 43  46 47 49 50 52 53    58 59 61 62  65 66 68 69 71 72 74 75  78 79      86 87  89 90  93     
#               -26        -3       0 -3     -2      -6      2       -4      1     -3     1     1      5       1      2     -1    1    -1     2     00000-1    -1      EOB
decompress(bitstream)