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