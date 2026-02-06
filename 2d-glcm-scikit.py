import glob, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile
from skimage import io, color,exposure,img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


image_path = '/home/woodn/Desktop/test.tif'

img = io.imread(image_path)
img_rescaled = exposure.rescale_intensity(img,
                                            out_range=(0, 255)).astype(np.uint8)

distance = [1,2,4,6,8]
angles = [0, np.pi/4,]

glcm_8bit = graycomatrix(img_rescaled,
                    distances = distance,
                    angles = angles,
                    levels = 256,
                    symmetric = True,
                    normed = True)

contrast = graycoprops(glcm_8bit, 'contrast')
homogeneity = graycoprops(glcm_8bit, 'homogeneity')
energy = graycoprops(glcm_8bit, 'energy')
corr = graycoprops(glcm_8bit, 'correlation')

print(f"Contrast: {np.mean(contrast):.2f}")
print(f"Homogeneity: {np.mean(homogeneity):.2f}")
print(f"Energy: {np.mean(energy):.2f}")
print(f"Correlation: {np.mean(corr):.2f}")
