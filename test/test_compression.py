import numpy as np
import pytest

from src.compression import *

class TestPadding:

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_0x0_image(self):
        img = np.zeros((0, 0))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (0, 0)
    
    def test_2x2_image(self):
        img = np.zeros((0, 0))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (0, 0)

    def test_8x8_image(self):
        img = np.ones((8, 8))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (8, 8)

    def test_9x9_image(self):
        img = np.ones((9, 9))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (16, 16)

    def test_11x3_image(self):
        img = np.ones((11, 3))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (16, 8)

    def test_3x11_image(self):
        img = np.ones((3, 11))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (8, 16)
    
    def test_16x3_image(self):
        img = np.ones((16, 3))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (16, 8)

    def test_3x16_image(self):
        img = np.ones((3, 16))
        for mode in ["black", "replicate"]:
            new_img = padding(img, mode=mode)
            assert new_img.shape == (8, 16)

class TestDct:
    @classmethod
    def setup_class(cls):
        # Taken from wikipedia: https://en.wikipedia.org/wiki/JPEG#Encoding
        cls.block = np.array([
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87,  79, 69, 68, 65, 76, 78, 94]]
        )

        cls.expected = np.array([
            [-415.38, -30.19, -61.20, 27.24, 56.12, -20.10, -2.39, 0.46],
            [4.47, -21.86, -60.76, 10.25, 13.15, -7.09, -8.54, 4.88],
            [-46.83, 7.37, 77.13, -24.56, -28.91, 9.93, 5.42, -5.65],
            [-48.53, 12.07, 34.10, -14.76, -10.24, 6.30, 1.83, 1.95],
            [12.12, -6.55, -13.20, -3.95, -1.87, 1.75, -2.79, 3.14],
            [-7.73, 2.91, 2.38, -5.94, -2.38, 0.94, 4.30, 1.85],
            [-1.03, 0.18, 0.42, -2.42, -0.88, -3.02, 4.12, -0.66],
            [-0.17, 0.14, -1.07, -4.19, -1.17, -0.10, 0.50, 1.68]]
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_dct(self):
        result = dct(self.block)
        np.allclose(result, self.expected)

class TestQuantization:
    @classmethod
    def setup_class(cls):
        # Taken from wikipedia: https://en.wikipedia.org/wiki/JPEG#Encoding
        cls.dct_block = np.array([
            [-415.38, -30.19, -61.20, 27.24, 56.12, -20.10, -2.39, 0.46],
            [4.47, -21.86, -60.76, 10.25, 13.15, -7.09, -8.54, 4.88],
            [-46.83, 7.37, 77.13, -24.56, -28.91, 9.93, 5.42, -5.65],
            [-48.53, 12.07, 34.10, -14.76, -10.24, 6.30, 1.83, 1.95],
            [12.12, -6.55, -13.20, -3.95, -1.87, 1.75, -2.79, 3.14],
            [-7.73, 2.91, 2.38, -5.94, -2.38, 0.94, 4.30, 1.85],
            [-1.03, 0.18, 0.42, -2.42, -0.88, -3.02, 4.12, -0.66],
            [-0.17, 0.14, -1.07, -4.19, -1.17, -0.10, 0.50, 1.68]]
        )
        
        cls.Q_MAT = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72,  92, 95, 98, 112, 100, 103, 99]]
        )
        
        cls.expected = np.array([
            [-26, -3, -6, 2, 2, -1, 0, 0],
            [0, -2, -4, 1, 1, 0, 0, 0],
            [-3, 1, 5, -1, -1, 0, 0, 0],
            [-3, 1, 2, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_quantization(self):
        result = quantization(self.dct_block, self.Q_MAT)
        np.allclose(result, self.expected)
        assert result.dtype == self.expected.dtype

class TestZigZag:
    @classmethod
    def setup_class(cls):
        # Taken from wikipedia: https://en.wikipedia.org/wiki/JPEG#Encoding
        cls.quantized_block = np.array([
            [-26, -3, -6, 2, 2, -1, 0, 0],
            [0, -2, -4, 1, 1, 0, 0, 0],
            [-3, 1, 5, -1, -1, 0, 0, 0],
            [-3, 1, 2, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]
        )

        cls.expected = np.array(
            [-26,
            -3, 0,
            -3, -2, -6,
            2, -4, 1, -3,
            1, 1, 5, 1, 2,
            -1, 1, -1, 2, 0, 0,
            0, 0, 0, -1, -1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0,
            0, 0,
            0]
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_zigzag(self):
        result = zigzag(self.quantized_block)
        np.allclose(result, np.trim_zeros(self.expected, trim='b'))

class TestHuffman:

    @classmethod
    def setup_class(cls):
        cls.zigzag_order = np.array([-26, -3, 0, -3, -2, -6, 2, -4, 1, -3, 1, 1, 5, 1, 2, -1, 1, -1, 2, 0, 0, 0, 0, 0, -1, -1, 0])
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))
        cls.expected = np.array(
            ["11000101",
            "0100",
            "11100100",
            "0101",
            "100001",
            "0110",
            "100011",
            "001",
            "0100",
            "001",
            "001",
            "100101",
            "001",
            "0110",
            "000",
            "001",
            "000",
            "0110",
            "11110100",
            "000",
            "1010"]
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_huffman(self):
        result = huffman(self.zigzag_order, self.largest_range)
        assert len(result) == len(self.expected)
        
        for res, exp in zip(result, self.expected):
            assert res == exp

class TestEntropyCoding:
    @classmethod
    def setup_class(cls):
        # Taken from wikipedia: https://en.wikipedia.org/wiki/JPEG#Encoding
        cls.quantized_block = np.array([
            [-26, -3, -6, 2, 2, -1, 0, 0],
            [0, -2, -4, 1, 1, 0, 0, 0],
            [-3, 1, 5, -1, -1, 0, 0, 0],
            [-3, 1, 2, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]
        )
        
        cls.zigzag_order = zigzag(cls.quantized_block)
        # JPEG coefficient coding category 15
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))

        cls.expected = np.array(
            ["11000101",
            "0100",
            "11100100",
            "0101",
            "100001",
            "0110",
            "100011",
            "001",
            "0100",
            "001",
            "001",
            "100101",
            "001",
            "0110",
            "000",
            "001",
            "000",
            "0110",
            "11110100",
            "000",
            "1010"]
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_entropy_coding(self):
        result = entropy_coding(self.quantized_block, self.largest_range)

        assert len(result) == len(self.expected)
        
        for res, exp in zip(result, self.expected):
            assert res == exp