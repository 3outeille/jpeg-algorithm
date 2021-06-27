import matplotlib.pyplot as plt
import os

from src.compression import *
from src.decompression import *
from src.utils import *

class TestPipeline:

    @classmethod
    def setup_class(cls):
        cls.largest_range = list(itertools.product(['0', '1'], repeat=15))

    @classmethod
    def teardown_class(cls):
        # os.remove("tmp.jpg")
        pass

    def test_pipeline(self):
        # img = plt.imread("../src/nyancat-patrick.png")
        # bitstream, unpadding_values = compression(self.img)
        # save_img(bitstream, "tmp.jpg")
        # load_bitstream = load_img("tmp.jpg")
        # decompress(load_bitstream, unpadding_values)
        pass

    def test_pipeline_1_block(self):
        expected = np.array([
            [62, 65, 57, 60, 72, 63, 60, 82],
            [57, 55, 56, 82, 108, 87, 62, 71],
            [58, 50, 60, 111, 148, 114, 67, 65],
            [65, 55, 66, 120, 155, 114, 68, 70],
            [70, 63, 67, 101, 122, 88, 60, 78],
            [71, 71, 64, 70, 80, 62, 56, 81],
            [75, 82, 67, 54, 63, 65, 66, 83],
            [81, 94, 75, 54, 68, 81, 81, 8]
        ])

        block = np.array([
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87,  79, 69, 68, 65, 76, 78, 94]]
        )

        # Compression

        dct_block = dct(block) # Step 2: Discrete cosine transform (DCT)
        q_block = quantization(dct_block, Q_MAT) # Step 3: Quantization + Round to nearest integer
        final_encoding = entropy_coding(q_block, self.largest_range) # Step 4: Zigzag + Huffman
        bitstream = "".join(map(str, np.concatenate([final_encoding])))

        # Decompression
        q_block_retrieved = entropy_coding_inv(bitstream, self.largest_range)
        dct_block_retrieved = quantization_inv(q_block_retrieved, Q_MAT)
        block_retrieved = dct_inv(dct_block_retrieved)

        np.allclose(block_retrieved, expected)
    
    def test_pipeline_1_block_with_depth(self):
        expected = np.array([
            [
                [62, 65, 57, 60, 72, 63, 60, 82],
                [57, 55, 56, 82, 108, 87, 62, 71],
                [58, 50, 60, 111, 148, 114, 67, 65],
                [65, 55, 66, 120, 155, 114, 68, 70],
                [70, 63, 67, 101, 122, 88, 60, 78],
                [71, 71, 64, 70, 80, 62, 56, 81],
                [75, 82, 67, 54, 63, 65, 66, 83],
                [81, 94, 75, 54, 68, 81, 81, 8]
            ],
            [
                [62, 65, 57, 60, 72, 63, 60, 82],
                [57, 55, 56, 82, 108, 87, 62, 71],
                [58, 50, 60, 111, 148, 114, 67, 65],
                [65, 55, 66, 120, 155, 114, 68, 70],
                [70, 63, 67, 101, 122, 88, 60, 78],
                [71, 71, 64, 70, 80, 62, 56, 81],
                [75, 82, 67, 54, 63, 65, 66, 83],
                [81, 94, 75, 54, 68, 81, 81, 8]
            ]
        ])

        block = np.array([
            [
                [52, 55, 61, 66, 70, 61, 64, 73],
                [63, 59, 55, 90, 109, 85, 69, 72],
                [62, 59, 68, 113, 144, 104, 66, 73],
                [63, 58, 71, 122, 154, 106, 70, 69],
                [67, 61, 68, 104, 126, 88, 68, 70],
                [79, 65, 60, 70, 77, 68, 58, 75],
                [85, 71, 64, 59, 55, 61, 65, 83],
                [87,  79, 69, 68, 65, 76, 78, 94]
            ],
            [
                [52, 55, 61, 66, 70, 61, 64, 73],
                [63, 59, 55, 90, 109, 85, 69, 72],
                [62, 59, 68, 113, 144, 104, 66, 73],
                [63, 58, 71, 122, 154, 106, 70, 69],
                [67, 61, 68, 104, 126, 88, 68, 70],
                [79, 65, 60, 70, 77, 68, 58, 75],
                [85, 71, 64, 59, 55, 61, 65, 83],
                [87,  79, 69, 68, 65, 76, 78, 94]
            ]
        ])

        result = []

        for channel in range(block.shape[0]):
            # Compression
            dct_block = dct(block[channel, ...]) # Step 2: Discrete cosine transform (DCT)
            q_block = quantization(dct_block, Q_MAT) # Step 3: Quantization + Round to nearest integer
            final_encoding = entropy_coding(q_block, self.largest_range) # Step 4: Zigzag + Huffman
            bitstream = "".join(map(str, np.concatenate([final_encoding])))
        
            # Decompression
            q_block_retrieved = entropy_coding_inv(bitstream, self.largest_range)
            dct_block_retrieved = quantization_inv(q_block_retrieved, Q_MAT)
            block_retrieved = dct_inv(dct_block_retrieved)

            result.append(block_retrieved)

        result = np.array(result)
    
        np.allclose(result, expected)

    def test_pipeline_2_block(self):
        blocks = np.array([
            [52, 55, 61, 66, 70, 61, 64, 73, 52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72, 63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73, 62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69, 63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70, 67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75, 79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83, 85, 71, 64, 59, 55, 61, 65, 83],
            [87,  79, 69, 68, 65, 76, 78, 94, 87,  79, 69, 68, 65, 76, 78, 94]
        ])

        expected = np.array([
            [62, 65, 57, 60, 72, 63, 60, 82, 62, 65, 57, 60, 72, 63, 60, 82],
            [57, 55, 56, 82, 108, 87, 62, 71, 57, 55, 56, 82, 108, 87, 62, 71],
            [58, 50, 60, 111, 148, 114, 67, 65, 58, 50, 60, 111, 148, 114, 67, 65],
            [65, 55, 66, 120, 155, 114, 68, 70, 65, 55, 66, 120, 155, 114, 68, 70],
            [70, 63, 67, 101, 122, 88, 60, 78, 70, 63, 67, 101, 122, 88, 60, 78],
            [71, 71, 64, 70, 80, 62, 56, 81, 71, 71, 64, 70, 80, 62, 56, 81],
            [75, 82, 67, 54, 63, 65, 66, 83, 75, 82, 67, 54, 63, 65, 66, 83],
            [81, 94, 75, 54, 68, 81, 81, 8, 81, 94, 75, 54, 68, 81, 81, 8]
        ])

        result = []

        for block in block_splitting(blocks):
            # Compression
            dct_block = dct(block) # Step 2: Discrete cosine transform (DCT)
            q_block = quantization(dct_block, Q_MAT) # Step 3: Quantization + Round to nearest integer
            final_encoding = entropy_coding(q_block, self.largest_range) # Step 4: Zigzag + Huffman
            bitstream = "".join(map(str, np.concatenate([final_encoding])))
        
            # Decompression
            q_block_retrieved = entropy_coding_inv(bitstream, self.largest_range)
            dct_block_retrieved = quantization_inv(q_block_retrieved, Q_MAT)
            block_retrieved = dct_inv(dct_block_retrieved)

            result.append(block_retrieved)

        result = np.concatenate(result, axis=1)
        np.allclose(result, expected)

    def test_pipeline_2_block_with_depth(self):
        blocks = np.array([
            [
                [52, 55, 61, 66, 70, 61, 64, 73, 52, 55, 61, 66, 70, 61, 64, 73],
                [63, 59, 55, 90, 109, 85, 69, 72, 63, 59, 55, 90, 109, 85, 69, 72],
                [62, 59, 68, 113, 144, 104, 66, 73, 62, 59, 68, 113, 144, 104, 66, 73],
                [63, 58, 71, 122, 154, 106, 70, 69, 63, 58, 71, 122, 154, 106, 70, 69],
                [67, 61, 68, 104, 126, 88, 68, 70, 67, 61, 68, 104, 126, 88, 68, 70],
                [79, 65, 60, 70, 77, 68, 58, 75, 79, 65, 60, 70, 77, 68, 58, 75],
                [85, 71, 64, 59, 55, 61, 65, 83, 85, 71, 64, 59, 55, 61, 65, 83],
                [87,  79, 69, 68, 65, 76, 78, 94, 87,  79, 69, 68, 65, 76, 78, 94]
            ],
            [
                [52, 55, 61, 66, 70, 61, 64, 73, 52, 55, 61, 66, 70, 61, 64, 73],
                [63, 59, 55, 90, 109, 85, 69, 72, 63, 59, 55, 90, 109, 85, 69, 72],
                [62, 59, 68, 113, 144, 104, 66, 73, 62, 59, 68, 113, 144, 104, 66, 73],
                [63, 58, 71, 122, 154, 106, 70, 69, 63, 58, 71, 122, 154, 106, 70, 69],
                [67, 61, 68, 104, 126, 88, 68, 70, 67, 61, 68, 104, 126, 88, 68, 70],
                [79, 65, 60, 70, 77, 68, 58, 75, 79, 65, 60, 70, 77, 68, 58, 75],
                [85, 71, 64, 59, 55, 61, 65, 83, 85, 71, 64, 59, 55, 61, 65, 83],
                [87,  79, 69, 68, 65, 76, 78, 94, 87,  79, 69, 68, 65, 76, 78, 94]
            ]
        ])

        expected = np.array([
            [
                [62, 65, 57, 60, 72, 63, 60, 82, 62, 65, 57, 60, 72, 63, 60, 82],
                [57, 55, 56, 82, 108, 87, 62, 71, 57, 55, 56, 82, 108, 87, 62, 71],
                [58, 50, 60, 111, 148, 114, 67, 65, 58, 50, 60, 111, 148, 114, 67, 65],
                [65, 55, 66, 120, 155, 114, 68, 70, 65, 55, 66, 120, 155, 114, 68, 70],
                [70, 63, 67, 101, 122, 88, 60, 78, 70, 63, 67, 101, 122, 88, 60, 78],
                [71, 71, 64, 70, 80, 62, 56, 81, 71, 71, 64, 70, 80, 62, 56, 81],
                [75, 82, 67, 54, 63, 65, 66, 83, 75, 82, 67, 54, 63, 65, 66, 83],
                [81, 94, 75, 54, 68, 81, 81, 8, 81, 94, 75, 54, 68, 81, 81, 8]
            ],
            [
                [62, 65, 57, 60, 72, 63, 60, 82, 62, 65, 57, 60, 72, 63, 60, 82],
                [57, 55, 56, 82, 108, 87, 62, 71, 57, 55, 56, 82, 108, 87, 62, 71],
                [58, 50, 60, 111, 148, 114, 67, 65, 58, 50, 60, 111, 148, 114, 67, 65],
                [65, 55, 66, 120, 155, 114, 68, 70, 65, 55, 66, 120, 155, 114, 68, 70],
                [70, 63, 67, 101, 122, 88, 60, 78, 70, 63, 67, 101, 122, 88, 60, 78],
                [71, 71, 64, 70, 80, 62, 56, 81, 71, 71, 64, 70, 80, 62, 56, 81],
                [75, 82, 67, 54, 63, 65, 66, 83, 75, 82, 67, 54, 63, 65, 66, 83],
                [81, 94, 75, 54, 68, 81, 81, 8, 81, 94, 75, 54, 68, 81, 81, 8]
            ]
        ])

        result = []

        for channel in range(blocks.shape[0]):

            tmp_channel = []

            for block in block_splitting(blocks[channel, ...]):
                # Compression
                dct_block = dct(block) # Step 2: Discrete cosine transform (DCT)
                q_block = quantization(dct_block, Q_MAT) # Step 3: Quantization + Round to nearest integer
                final_encoding = entropy_coding(q_block, self.largest_range) # Step 4: Zigzag + Huffman
                bitstream = "".join(map(str, np.concatenate([final_encoding])))
            
                # Decompression
                q_block_retrieved = entropy_coding_inv(bitstream, self.largest_range)
                dct_block_retrieved = quantization_inv(q_block_retrieved, Q_MAT)
                block_retrieved = dct_inv(dct_block_retrieved)
                tmp_channel.append(block_retrieved)

            result.append(tmp_channel)

        result = np.concatenate(result, axis=2)
        np.allclose(result, expected)

    
    def test_pipeline_with_image(self):
        img = plt.imread("../src/square.png")[..., :3]        
        img = np.transpose(img, (2, 0, 1))
        unpadding_values = {
            "ax1_top": [],
            "ax1_bot": [],
            "ax2_left": [], 
            "ax2_right": []
        }

        result = []

        for channel in range(3):

            img_channel = padding(img[channel, ...], unpadding_values, mode="replicate")
            
            tmp_channel = []

            for block in block_splitting(img_channel):
                print(img_channel.shape)
                plt.imshow(img_channel)
                raise Exception("")
                # Compression
                dct_block = dct(block) # Step 2: Discrete cosine transform (DCT)
                q_block = quantization(dct_block, Q_MAT) # Step 3: Quantization + Round to nearest integer
                final_encoding = entropy_coding(q_block, self.largest_range) # Step 4: Zigzag + Huffman
                bitstream = "".join(map(str, np.concatenate([final_encoding])))
            
                # Decompression
                q_block_retrieved = entropy_coding_inv(bitstream, self.largest_range)
                dct_block_retrieved = quantization_inv(q_block_retrieved, Q_MAT)
                block_retrieved = dct_inv(dct_block_retrieved)
                tmp_channel.append(block_retrieved)

            result.append(tmp_channel)

        result = np.concatenate(result, axis=2)


        pass