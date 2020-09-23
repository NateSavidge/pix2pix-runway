# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from post_process import Quantize
from datetime import datetime

class Pix2Pix():

    def __init__(self, options):
        checkpoint_path = options['checkpoint']

        if checkpoint_path is not None: 
            self.model = tf.keras.models.load_model(checkpoint_path)
        else:
            ## With walls:
            # self.model = tf.keras.models.load_model('generator_model_001_floorplanswalls_062320.h5')
            ## No walls:
            self.model = tf.keras.models.load_model('generator_model_003_061720.h5')


    # Generate an image based on a 256 by 256 input shape:
    def run_on_input(self, input_image):
        input_image_processed = tf.image.convert_image_dtype(np.array(input_image), tf.float32)
        input_image_processed = tf.image.resize(input_image_processed, (256, 256), antialias=True)
        input_image_processed = tf.expand_dims(input_image_processed, axis=[0]) # (1, 256, 256, 3)
        prediction = self.model(input_image_processed, training=True)

        output_image = prediction[0] # -> (256, 256, 3)

        array_from_tensor = np.asarray(output_image)

        # Remap the pixel values from float values to integer values
        mapped_array = np.interp(array_from_tensor, (0, +1), (0, +255))

        # Construct a PIL image to display in Runway
        PIL_IMG = Image.fromarray(np.uint8(mapped_array))
        #return PIL_IMG
        # # Quantize image (if quantize=true)
        quantize = Quantize()
        quantized = quantize.from_model(PIL_IMG)

        img_name_1 = 'orig-' + datetime.now().strftime("%H%M%S") + '.png'
        img_name_2 = 'quantized-' + datetime.now().strftime("%H%M%S") + '.png'
        print(img_name_1)

        PIL_IMG.save(img_name_1)
        quantized.save(img_name_2)

        return quantized