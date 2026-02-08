'''
Nathan Wood 2026 contact@nwoodweb.xyz

MIT LICENSE

---------------
RETURNS
---------------

---------------
FUNCTIONS
---------------

---------------
USER PARAMETERS
---------------

lateral_pixel_size: float64

axial_pixel_size: float64

'''


import numpy as np
from scipy.ndimage import zoom


# USER DEFINED

lateral_pixel_size = 0.052  # units micron
axial_pixel_size = 0.18     # units micron
box_size = 64   #units pixel
overlap = 0.5   #unit [0,1]



def global_float64_conversion(zstack):
    zstack_float64 = zstack.astype(np.float64) / 65535.0
    return zstack_float64

def hann_window(size)

    window_init  = np.hanning(size)
    window_2d = np.outer(window, window)
    window_3d = np.outer(window_init, window_2d).reshape(size,size,size) 

    return window_3d

def rescale_voxels(zstack):

    z_rescale_factor = axial_pixel_size / lateral_pixel_size

    zstack_rescaled = zoom(zstack, (z_rescale_factor, 1, 1),
                           order = 1)

    return zstack_rescaled

