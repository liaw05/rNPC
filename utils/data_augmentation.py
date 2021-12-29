#coding=utf-8
import numpy as np


def random_swap_xy(data, mask=None, centerd=None, bbox=None):
    #data: zyx
    if data.shape[1] == data.shape[2]:
        axisorder = np.random.permutation([1,2])   #将维度顺序[1,2]打乱
        axisorder = np.insert(axisorder, 0, 0)
        data = np.transpose(data, axisorder)
        if mask is not None:
            mask = np.transpose(mask, axisorder)
        if centerd is not None:
            centerd[:, :3] = centerd[:, :3][:, axisorder]
            centerd[:, 3:6] = centerd[:, 3:6][:, axisorder]
        if bbox is not None:
            bbox[:, :3] = bbox[:, :3][:, axisorder]
            bbox[:, 3:6] = bbox[:, 3:6][:, axisorder]

    ret = [data, mask, centerd, bbox]
    ret = [ele for ele in ret if ele is not None]
    return ret


def random_flip_func(data, mask=None, centerd=None, bbox=None, axis=[0,1,2]):
    #data: zyx
    if np.random.uniform(0,1)>0.5:
        for ii in axis:
            if np.random.randint(0, 2):
                data = np.flip(data, ii)
                if mask is not None:
                    mask = np.flip(mask, ii)
                if centerd is not None:
                    centerd[:, ii] = data.shape[ii] - 1 - centerd[:, ii]
                if bbox is not None:
                    bbox[:, [ii, ii+3]] = bbox[:, [ii+3, ii]]
                    bbox[:, ii] = data.shape[ii] - 1 - bbox[:, ii]
                    bbox[:, ii + 3] = data.shape[ii] - 1 - bbox[:, ii + 3]

    ret = [data, mask, centerd, bbox]
    ret = [ele for ele in ret if ele is not None]
    return ret


def random_crop_pad_func(image, crop_size, centerd=None, return_croppad_id=True, 
                         pad_value=0, is_random=True):
    # pad to shape >= crop size
    image_shape = image.shape
    start_zyx = np.array([0, 0, 0])
    if np.any([image_shape[dim] < ps for dim, ps in enumerate(crop_size)]):
        new_shape = [max(image_shape[i], crop_size[i]) for i in range(len(crop_size))]
        difference = np.array(new_shape) - image_shape
        pad_below = difference // 2
        pad_above = difference-pad_below
        start_zyx = start_zyx - pad_below
        pad_list = [list(i) for i in zip(pad_below, pad_above)]
        image = np.pad(image, pad_list, mode='constant', constant_values=pad_value)
        if centerd is not None:
            centerd[:,:3] = centerd[:,:3] + np.array(pad_below)

    # crop to crop size
    image_shape = image.shape
    for ii in range(len(image_shape)):
        diff = image_shape[ii]-crop_size[ii]
        if diff>0:
            if is_random:
                min_crop = np.random.randint(0, diff)
            else:
                min_crop = diff//2
            max_crop = min_crop + crop_size[ii]
            start_zyx[ii] = start_zyx[ii] + min_crop
            image = np.take(image, indices=range(min_crop, max_crop), axis=ii)
            if centerd is not None:
                centerd[:, ii] -= min_crop

    if return_croppad_id:
        return image, centerd, start_zyx
    else:
        return image, centerd
