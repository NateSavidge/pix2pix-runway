import numpy as np
import scipy
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from time import time
from PIL import Image

## Sklearn package issues - this finally fixed it (in conda py36 environment)
## "pip3 install -U scikit-learn scipy matplotlib"
class Quantize():

    # input_image = Image.open('test/01.jpeg')
    def __init__(self):
        self.colors = [255, 0, 255,
                       255, 255, 51, 
                       255, 0, 0, 
                       51, 255, 255, 
                       0, 255, 0, 
                       255, 153, 0, 
                       0, 0, 255, 
                       255, 255, 255, 
                       0, 0, 0]

    def recreate_image(self, codebook, labels, w, h):
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def from_model(self, input_image):
        n_colors = 9
        # 7 Standard tag colors + black + white
        colors = self.colors
        # colors_RGB = np.reshape(colors, (9,3))
        colors_float = np.array(colors, dtype=np.float64) / 255
        
        # Downsample image before quantizing:
        # input_image = input_image.resize((32,32), resample=0)
        # input_image = input_image.resize((64,64), resample=Image.NEAREST)

        input_image = np.array(input_image, dtype=np.float64) / 255
        w, h, d, = tuple(input_image.shape)
        print(w, h, d)
        assert d == 3

        image_array = np.reshape(input_image, (w*h, d))
        # Split array into subarrays of [R,G,B] values
        colors_array = np.reshape(colors_float, (n_colors, 3))

        # KMeans V2:
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(colors_array)
        labels = kmeans.predict(image_array)
        output_image = self.recreate_image(kmeans.cluster_centers_, labels, w, h)
        
        # Pairwise Distances Method:
        # labels_fixed = pairwise_distances_argmin(colors_array, image_array, axis=0)
        # output_image = self.recreate_image(colors_array, labels_fixed, w, h)
        
        # Convert back to integers from floats:
        as_int = (output_image * 255).astype(np.uint8)

        # output_image = Image.fromarray(as_int, mode='RGB').resize((256,256), resample=Image.NONE)
        output_image = Image.fromarray(as_int, mode='RGB').resize((64,64), resample=Image.NEAREST)

        # output_image = output_image.resize((64,64), resample=Image.NONE)
        # output_image = Image.fromarray(as_int, mode='RGB')
        
        return(output_image)