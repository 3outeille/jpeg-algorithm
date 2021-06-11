import numpy as np
import scipy as sp

from utils import Q_MAT, HUFFMAN_AC_TABLE

def padding(img, mode="black"):
    """
        Only work for 2d matrix.
    """
    # Round up to nearest multiple of 8
    w = (img.shape[0] + 7) & (-8)
    h = (img.shape[1] + 7) & (-8)

    # Smart padding -> Pad with image center in middle.
    delta_w = w - img.shape[0]
    delta_h = h - img.shape[1]
    
    ax1_top, ax1_bot, ax2_left, ax2_right = 0, 0, 0, 0

    if delta_w != 0:
        ax1_top, ax1_bot = (delta_w//2) , (delta_w//2)
        if delta_w % 2 == 1:    
            ax1_bot = (delta_w//2) + 1
    
    if delta_h != 0:
        ax2_left, ax2_right = (delta_h//2) , (delta_h//2)
        if delta_h % 2 == 1:
            ax2_right = (delta_h//2) + 1

    if mode == "black":
        img = np.pad(img, [(ax1_top , ax1_bot), (ax2_left, ax2_right)], 'constant')
    elif mode == "replicate":
        img = np.pad(img, [(ax1_top, ax1_bot), (ax2_left, ax2_right)], 'symmetric')
    else:
        raise ValueError("This mode doesn't exist")

    return img

def block_splitting(img):
    """
        8x8 sliding window over image

        TODO: yield macro block later ?
    """
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            yield img[i:i+8, j:j+8]

def dct(block):
    block =  block - 128
    dct_block = sp.fft.dct(block, axis=0, type=2, norm="ortho")
    dct_block = sp.fft.dct(dct_block, axis=1, type=2, norm="ortho")
    return dct_block

def quantization(dct_block, Q):
    return np.rint(np.divide(dct_block, Q)).astype(int)

def zigzag(q_block):
    n, m = q_block.shape
    res = [[] for i in range(n + m - 1)]
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0:
                res[i + j].insert(0, q_block[i][j])
            else:
                res[i + j].append(q_block[i][j])

    return np.trim_zeros(np.concatenate(res), trim='b')
     
def compression(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image dimension is not supported")

    img = np.transpose(img, (2, 0, 1))
    
    img_compressed = []
    for channel in range(3):
        # Step 1: Block splitting
        img_channel = padding(img[channel, ...])

        for block in block_splitting(img_channel):
            # Step 2: Discrete cosine transform (DCT)
            dct_block = dct(block)
            # Step 3: Quantization + Round to nearest integer
            q_block = quantization(dct_block, Q_MAT)
            # Step 4: Zigzag + Huffman
            raise Exception("")

            img_compressed.append(block)
    
    return img_compressed

# dummy = np.ones((11, 11, 3))
# compression(dummy)