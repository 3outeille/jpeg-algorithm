import itertools
import numpy as np
import pytest

from src.decompression import *

class TestHuffmanInv:

    @classmethod
    def setup_class(cls):
        cls.bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))
        cls.expected = [-26, -3, 0, -3, -2, -6, 2, -4, 1, -3, 1, 1, 5, 1, 2, -1, 1, -1, 2, 0, 0, 0, 0, 0, -1, -1, 0]

    @classmethod
    def teardown_class(cls):
        pass

    def test_huffman_inv(self):
        result = huffman_inv(self.bitstream, self.largest_range)
        assert len(result) == len(self.expected)
        np.allclose(result, self.expected)

class TestZigzagInv:
    
    @classmethod
    def setup_class(cls):
        cls.final_encoding = [-26, -3, 0, -3, -2, -6, 2, -4, 1, -3, 1, 1, 5, 1, 2, -1, 1, -1, 2, 0, 0, 0, 0, 0, -1, -1, 0]
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

    def test_zigzag_inv(self):
        result = zigzag_inv(self.final_encoding)
        np.allclose(result, self.expected)


class TestEntropyCodingInv:
    
    @classmethod
    def setup_class(cls):
        cls.bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))
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

    def test_entropy_coding_inv(self):
        result = entropy_coding_inv(self.bitstream, self.largest_range)
        np.allclose(result, self.expected)

class TestQuantizationInv:

    @classmethod
    def setup_class(cls):
        cls.q_block = np.array([
            [-26, -3, -6, 2, 2, -1, 0, 0],
            [0, -2, -4, 1, 1, 0, 0, 0],
            [-3, 1, 5, -1, -1, 0, 0, 0],
            [-3, 1, 2, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]
        )

        cls.expected = np.array([
            [-416,  -33,  -60,   32,   48,  -40,    0,    0],
            [   0,  -24,  -56,   19,   26,    0,    0,    0],
            [ -42,   13,   80,  -24,  -40,    0,    0,    0],
            [ -42,   17,   44,  -29,    0,    0,    0,    0],
            [  18,    0,    0,    0,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,    0,    0,    0]
        ])

    @classmethod
    def teardown_class(cls):
        pass

    def test_quantization_inv(self):
        result = quantization_inv(self.q_block, Q_MAT)
        np.allclose(result, self.expected)