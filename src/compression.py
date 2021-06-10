import numpy as np
from numpy.core.fromnumeric import compress

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

def compression(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image dimension is not supported")

    img = np.transpose(img, (2, 0, 1))
    
    for channel in range(3):
        # Step 1: Block splitting
        img_channel = padding(img[channel, ...])
        print(img_channel.shape)
        for macro_block in block_splitting(img_channel):
            print(f"channel {channel}: {macro_block.shape}")
        # Step 2: DCT
        # Step 3: Quantization
        # Step 4: Zigzag + Huffman
    pass

# dummy = np.ones((11, 5, 3))
# compression(dummy)