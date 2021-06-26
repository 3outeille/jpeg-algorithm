import os

from src.decompression import *
from src.utils import *

class TestIO:

    @classmethod
    def setup_class(cls):
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))
        cls.bitstream = "1100010101001110010001011000010110100011001010000100110010100101100000010000110111101000001010"
        cls.expected = [-26, -3, 0, -3, -2, -6, 2, -4, 1, -3, 1, 1, 5, 1, 2, -1, 1, -1, 2, 0, 0, 0, 0, 0, -1, -1]

    @classmethod
    def teardown_class(cls):
        os.remove("tmp.jpg")

    def test_save_load(self):
        save_img(self.bitstream, "tmp.jpg")
        load_bitstream = load_img("tmp.jpg")

        assert len(self.bitstream) == len(load_bitstream)
        assert self.bitstream == load_bitstream

        result = huffman_inv(load_bitstream, self.largest_range)
        assert np.allclose(result, self.expected) == True