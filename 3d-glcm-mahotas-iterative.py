import os
from glob import glob
import pandas as pd
import numpy as np
import tifffile
from skimage import io, color,exposure,img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

import mahotas.features 

def glcm_automate_8bit(img):
        img_load = io.imread(img)
        img_clip = np.clip(img_load,0,6000)
        img_scale = ((img_clip - 0) / (6000-0)) * 255
        img_uint8 = img_scale.astype(np.uint8)
        features = mahotas.features.haralick(img_uint8)
        mean_features = features.mean(axis=0)

        angular_moment = mean_features[0]
        contrast = mean_features[1]
        correlation = mean_features[2]
        variance = mean_features[3]
        inverse_diff_moment = mean_features[4]
        entropy = mean_features[8]
        print(img)
        data = [img,angular_moment, contrast, correlation, variance,
                inverse_diff_moment, entropy]
        dataframe.append(data)

def glcm_automate_16bit(img):
        img_load = io.imread(img)
        img_rescaled = exposure.rescale_intensity(img_load,
                                              out_range=(0,65536)).astype(np.uint16)
        features = mahotas.features.haralick(img_rescaled)
        mean_features = features.mean(axis=0)

        angular_moment = mean_features[0]
        contrast = mean_features[1]
        correlation = mean_features[2]
        variance = mean_features[3]
        inverse_diff_moment = mean_features[4]
        entropy = mean_features[8]

        data = [img, angular_moment, contrast, correlation, variance,
                inverse_diff_moment, entropy]
        dataframe.append(data)


#image_path = '/home/woodn/Desktop/test-12-stack.tif'
image_path = '/home/woodn/Desktop/test/'

dataframe = []
for img in sorted(glob(os.path.join(image_path, "*.tif"))):
    glcm_automate_8bit(img = img)

dataframe = pd.DataFrame(dataframe)

dataframe_headers =  ['filename','angular_moment', 'contrast',
                      'correlation', 'variance', 'inverse_diff_moment',
                      'entropy']
dataframe.columns = dataframe_headers


dataframe.to_csv('/home/woodn/Desktop/test/out-01-5feb26.csv', sep = ',',
                 encoding = 'utf-8')
