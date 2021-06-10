import numpy as np
import pytest

from src.compression import padding

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