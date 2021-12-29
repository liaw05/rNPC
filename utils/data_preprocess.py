#encoding=utf-8
import numpy as np
from scipy.ndimage import zoom


def resample_image(image, sp_scale, centerd=None, rd_scale=(0.7,1.3), order=1, is_random=True):
    if is_random:
        sp_scale = np.array([np.random.uniform(rd_scale[0], rd_scale[1]) for _ in range(len(sp_scale))]) * sp_scale
        sp_scale = sp_scale.tolist()
    else:
        sp_scale = list(sp_scale)

    image = zoom(image, sp_scale, order=order)
    if centerd is not None:
        centerd[:, :6] *= np.array(sp_scale * 2)
    return image, centerd