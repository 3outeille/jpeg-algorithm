import itertools
import numpy as np

Q_MAT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72,  92, 95, 98, 112, 100, 103, 99]]
)

# Table A
# def jpeg_coefficient_coding_categories(x):
#     if x == 0:
#         return 0, [('0')]
#     else:
#         category = len(bin(x).partition('b')[-1])
#         return category, list(itertools.product(['0', '1'], repeat=category))

# def retrieve_binary_rep(L, x):
#     mid = len(L)//2
#     val = L[:mid][(x % mid) - 1] if (x < 0) else L[mid:][x - mid]
#     return "".join(val)

def decimal_to_binary(x, largest_range):
    """
        Get binary represetation from decimal given a category range.

        Notes:
            Optimized version compare to previous one.
            Only need to compute jpeg_coefficient_coding_categories once (category 15).
            We then take range from it.
    """
    category = len(bin(x).partition('b')[-1])
    subset = largest_range[0:2**category]
    mid = len(subset) // 2
    val = subset[:mid][(x % mid) - 1] if (x < 0) else subset[mid:][x - mid]
    return category, "".join(val)[-category:]

# largest_range = list(itertools.product(['0', '1'], repeat=15))
# print(retrieve_binary_rep(largest_range, -7))

def binary_to_decimal(binary, largest_range):
    CAT = len(binary)
    subset = largest_range[0:2**CAT]
    n = len(subset)
    mid, left = n // 2, n - 1

    for inc_right, elt in enumerate(subset):
        tmp = "".join(elt[-CAT:])
        if binary == tmp:
            return -left + inc_right if (inc_right < mid) else inc_right
    return 0

# largest_range = list(itertools.product(['0', '1'], repeat=15))
# print(binary_to_decimal("00101", largest_range))
# print(binary_to_decimal("01", largest_range))
# print(binary_to_decimal("", largest_range))

def save_img(bitsteam, filename):
    with open(filename, "wb") as f:
        tmp = [int(bitsteam[i:i+8], 2) for i in range(0, len(bitsteam), 8)]
        f.write(bytearray(tmp))

# x = -3
# print(x)
# CAT, L = jpeg_coefficient_coding_categories(x)
# print(CAT, len(L))
# print(retrieve_binary_rep(L, x))


# Table B
HUFFMAN_DC_TABLE = {
    0: "010",
    1: "011",
    2: "100",
    3: "00",
    4: "101",
    5: "110",
    6: "1110",
    7: "11110",
    8: "111110",
    9: "1111110",
    10: "11111110",
    11: "111111110"
}

HUFFMAN_DC_TABLE_INV = {
    "010": 0,
    "011": 1,
    "100": 2,
    "00": 3,
    "101": 4,
    "110": 5,
    "1110": 6,
    "11110": 7,
    "111110": 8,
    "1111110": 9,
    "11111110": 10,
    "111111110": 11
}

# Table C
HUFFMAN_AC_TABLE = {
    '0/0': "1010",
    '0/1': "00",
    '0/2': "01",
    '0/3': "100",
    '0/4': "1011",
    '0/5': "11010",
    '0/6': "111000",
    '0/7': "1111000",
    '0/8': "1111110110",
    '0/9': "1111111110000010",
    '0/10':"1111111110000011",
    '1/1': "110",
    '1/2': "111001",
    '1/3': "1111001",
    '1/4': "111110110",
    '1/5': "11111110110",
    '1/6': "1111111110000100",
    '1/7': "1111111110000101",
    '1/8': "1111111110000110",
    '1/9': "1111111110000111",
    '1/10':"1111111110001000",
    '2/1': "11011",
    '2/2': "11111000",
    '2/3': "1111110111",
    '2/4': "1111111110001001",
    '2/5': "1111111110001010",
    '2/6': "1111111110001011",
    '2/7': "1111111110001100",
    '2/8': "1111111110001101",
    '2/9': "1111111110001110",
    '2/10':"1111111110001111",
    '3/1': "111010",
    '3/2': "111110111",
    '3/3': "11111110111",
    '3/4': "1111111110010000",
    '3/5': "1111111110010001",
    '3/6': "1111111110010010",
    '3/7': "1111111110010011",
    '3/8': "11111111110010100",
    '3/9': "11111111110010101",
    '3/10': "11111111110010110",
    '4/1': "111011",
    '4/2': "1111111000",
    '4/3': "1111111110010111",
    '4/4': "1111111110011000",
    '4/5': "1111111110011001",
    '4/6': "1111111110011010",
    '4/7': "1111111110011011",
    '4/8': "1111111110011100",
    '4/9': "1111111110011101",
    '4/10':"1111111110011110",
    '5/1': "1111010",
    '5/2': "1111111001",
    '5/3': "1111111110011111",
    '5/4': "1111111110100000",
    '5/5': "1111111110100001",
    '5/6': "1111111110100010",
    '5/7': "1111111110100011",
    '5/8': "1111111110100100",
    '5/9': "1111111110100101",
    '5/10':"1111111110100110" ,
    '6/1': "1111011",
    '6/2': "11111111000",
    '6/3': "1111111110100111",
    '6/4': "1111111110101000",
    '6/5': "1111111110101001",
    '6/6': "1111111110101010",
    '6/7': "1111111110101011",
    '6/8': "1111111110101100",
    '6/9': "1111111110101101",
    '6/10':"1111111110101110",
    '7/1': "11111001",
    '7/2': "11111111001",
    '7/3': "1111111110101111",
    '7/4': "1111111110110000",
    '7/5': "1111111110110001",
    '7/6': "1111111110110010",
    '7/7': "1111111110110011",
    '7/8': "1111111110110100",
    '7/9': "1111111110110101",
    '7/10':"1111111110110110",
    '8/1': "11111010",
    '8/2': "111111111000000",
    '8/3': "1111111110110111",
    '8/4': "1111111110111000",
    '8/5': "1111111110111001",
    '8/6': "1111111110111010",
    '8/7': "1111111110111011",
    '8/8': "1111111110111100",
    '8/9': "1111111110111101",
    '8/10':"1111111110111110",
    '9/1': "111111000",
    '9/2': "1111111110111111",
    '9/3': "1111111111000000",
    '9/4': "1111111111000001",
    '9/5': "1111111111000010",
    '9/6': "1111111111000011",
    '9/7': "1111111111000100",
    '9/8': "1111111111000101",
    '9/9': "1111111111000110",
    '9/10':"1111111111000111",
    '10/1':"111111001",
    '10/2':"1111111111001000",
    '10/3':"1111111111001001",
    '10/4':"1111111111001010",
    '10/5':"1111111111001011",
    '10/6':"1111111111001100",
    '10/7':"1111111111001101",
    '10/8':"1111111111001110",
    '10/9':"1111111111001111",
    '10/10':"1111111111010000",
    '11/1':"111111010",
    '11/2':"1111111111010001",
    '11/3':"1111111111010010",
    '11/4':"1111111111010011",
    '11/5':"1111111111010100",
    '11/6':"1111111111010101",
    '11/7':"1111111111010110",
    '11/8':"1111111111010111",
    '11/9':"1111111111011000",
    '11/10':"1111111111011001",
    '12/1':"1111111010",
    '12/2':"1111111111011010",
    '12/3':"1111111111011011",
    '12/4':"1111111111011100",
    '12/5':"1111111111011101",
    '12/6':"1111111111011110",
    '12/7':"1111111111011111",
    '12/8':"1111111111100000",
    '12/9':"1111111111100001",
    '12/10':"1111111111100010",
    '13/1':"11111111010",
    '13/2':"1111111111100011",
    '13/3':"1111111111100100",
    '13/4':"1111111111100101",
    '13/5':"1111111111100110",
    '13/6':"1111111111100111",
    '13/7':"1111111111101000",
    '13/8':"1111111111101001",
    '13/9':"1111111111101010",
    '13/10':"1111111111101011",
    '14/1':"111111110110",
    '14/2':"1111111111101100",
    '14/3':"1111111111101101",
    '14/4':"1111111111101110",
    '14/5':"1111111111101111",
    '14/6':"1111111111110000",
    '14/7':"1111111111110001",
    '14/8':"1111111111110010",
    '14/9':"1111111111110011",
    '14/10':"1111111111110100",
    '15/0':"111111110111",
    '15/1':"1111111111110101",
    '15/2':"1111111111110110",
    '15/3':"1111111111110111",
    '15/4':"1111111111111000",
    '15/5':"1111111111111001",
    '15/6':"1111111111111010",
    '15/7':"1111111111111011",
    '15/8':"1111111111111100",
    '15/9':"1111111111111101",
    '15/10':"1111111111111110"
}

# for run in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']:
#     for category in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
#         print(f"{run}/{category}")

HUFFMAN_AC_TABLE_INV = {
    "1010": "0/0",
    "00": "0/1",
    "01": "0/2",
    "100": "0/3",
    "1011": "0/4",
    "11010": "0/5",
    "111000": "0/6",
    "1111000": "0/7",
    "1111110110": "0/8",
    "1111111110000010": "0/9",
    "1111111110000011": "0/10",
    "110": "1/1",
    "111001": "1/2",
    "1111001": "1/3",
    "111110110": "1/4",
    "11111110110": "1/5",
    "1111111110000100": "1/6",
    "1111111110000101": "1/7",
    "1111111110000110": "1/8",
    "1111111110000111": "1/9",
    "1111111110001000": "1/10",
    "11011": "2/1",
    "11111000": "2/2",
    "1111110111": "2/3",
    "1111111110001001": "2/4",
    "1111111110001010": "2/5",
    "1111111110001011": "2/6",
    "1111111110001100": "2/7",
    "1111111110001101": "2/8",
    "1111111110001110": "2/9",
    "1111111110001111": "2/10",
    "111010": "3/1",
    "111110111": "3/2",
    "11111110111": "3/3",
    "1111111110010000": "3/4",
    "1111111110010001": "3/5",
    "1111111110010010": "3/6",
    "1111111110010011": "3/7",
    "11111111110010100": "3/8",
    "11111111110010101": "3/9",
    "11111111110010110": "3/10",
    "111011": "4/1",
    "1111111000": "4/2",
    "1111111110010111": "4/3",
    "1111111110011000": "4/4",
    "1111111110011001": "4/5",
    "1111111110011010": "4/6",
    "1111111110011011": "4/7",
    "1111111110011100": "4/8",
    "1111111110011101": "4/9",
    "1111111110011110": "4/10",
    "1111010": "5/1",
    "1111111001": "5/2",
    "1111111110011111": "5/3",
    "1111111110100000": "5/4",
    "1111111110100001": "5/5",
    "1111111110100010": "5/6",
    "1111111110100011": "5/7",
    "1111111110100100": "5/8",
    "1111111110100101": "5/9",
    "1111111110100110": "5/10",
    "1111011": "6/1",
    "11111111000": "6/2",
    "1111111110100111": "6/3",
    "1111111110101000": "6/4",
    "1111111110101001": "6/5",
    "1111111110101010": "6/6",
    "1111111110101011": "6/7",
    "1111111110101100": "6/8",
    "1111111110101101": "6/9",
    "1111111110101110": "6/10",
    "11111001": "7/1",
    "11111111001": "7/2",
    "1111111110101111": "7/3",
    "1111111110110000": "7/4",
    "1111111110110001": "7/5",
    "1111111110110010": "7/6",
    "1111111110110011": "7/7",
    "1111111110110100": "7/8",
    "1111111110110101": "7/9",
    "1111111110110110": "7/10",
    "11111010": "8/1",
    "111111111000000": "8/2",
    "1111111110110111": "8/3",
    "1111111110111000": "8/4",
    "1111111110111001": "8/5",
    "1111111110111010": "8/6",
    "1111111110111011": "8/7",
    "1111111110111100": "8/8",
    "1111111110111101": "8/9",
    "1111111110111110": "8/10",
    "111111000": "9/1",
    "1111111110111111": "9/2",
    "1111111111000000": "9/3",
    "1111111111000001": "9/4",
    "1111111111000010": "9/5",
    "1111111111000011": "9/6",
    "1111111111000100": "9/7",
    "1111111111000101": "9/8",
    "1111111111000110": "9/9",
    "1111111111000111": "9/10",
    "111111001": "10/1",
    "1111111111001000": "10/2",
    "1111111111001001": "10/3",
    "1111111111001010": "10/4",
    "1111111111001011": "10/5",
    "1111111111001100": "10/6",
    "1111111111001101": "10/7",
    "1111111111001110": "10/8",
    "1111111111001111": "10/9",
    "1111111111010000": "10/10",
    "111111010": "11/1",
    "1111111111010001": "11/2",
    "1111111111010010": "11/3",
    "1111111111010011": "11/4",
    "1111111111010100": "11/5",
    "1111111111010101": "11/6",
    "1111111111010110": "11/7",
    "1111111111010111": "11/8",
    "1111111111011000": "11/9",
    "1111111111011001": "11/10",
    "1111111010": "12/1",
    "1111111111011010": "12/2",
    "1111111111011011": "12/3",
    "1111111111011100": "12/4",
    "1111111111011101": "12/5",
    "1111111111011110": "12/6",
    "1111111111011111": "12/7",
    "1111111111100000": "12/8",
    "1111111111100001": "12/9",
    "1111111111100010": "12/10",
    "11111111010": "13/1",
    "1111111111100011": "13/2",
    "1111111111100100": "13/3",
    "1111111111100101": "13/4",
    "1111111111100110": "13/5",
    "1111111111100111": "13/6",
    "1111111111101000": "13/7",
    "1111111111101001": "13/8",
    "1111111111101010": "13/9",
    "1111111111101011": "13/10",
    "111111110110": "14/1",
    "1111111111101100": "14/2",
    "1111111111101101": "14/3",
    "1111111111101110": "14/4",
    "1111111111101111": "14/5",
    "1111111111110000": "14/6",
    "1111111111110001": "14/7",
    "1111111111110010": "14/8",
    "1111111111110011": "14/9",
    "1111111111110100": "14/10",
    "111111110111": "15/0",
    "1111111111110101": "15/1",
    "1111111111110110": "15/2",
    "1111111111110111": "15/3",
    "1111111111111000": "15/4",
    "1111111111111001": "15/5",
    "1111111111111010": "15/6",
    "1111111111111011": "15/7",
    "1111111111111100": "15/8",
    "1111111111111101": "15/9",
    "1111111111111110": "15/10"
}

# Generate HUFFMAN_AC_TABLE_INV table
# with open("tmp.txt", "w") as f:
#     for i, (key, val) in enumerate(HUFFMAN_AC_TABLE.items()):
#         f.write(f"\"{val}\": \"{key}\"")
#         if i < len(HUFFMAN_AC_TABLE) - 1:
#             f.write(",\n")
#         else:
#             f.write("\n")