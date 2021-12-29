
import torch.utils.data as data
import torch
import numpy as np
import pickle
import copy
from skimage import exposure
import SimpleITK as sitk
from scipy.ndimage.interpolation import rotate,zoom
import warnings
warnings.filterwarnings("ignore")


class DataCustom(data.Dataset):
    '''
    NPC数据生成器
    :param state: train or val or test
    :param train_img_root: 训练集img的文件夹路径
    :param val_img_root: 验证集img的文件夹路径
    :param test_img_root: 测试集img的文件夹路径
    :param train_mask_root: 训练集mask的文件夹路径
    :param val_mask_root: 验证集mask的文件夹路径
    :param test_mask_root: 测试集mask的文件夹路径
    '''
    def __init__(self, state, cf):
        self.state = state
        self.input_size = cf.input_size

        self.data_train_val_test_20path = cf.train_val_test20_data_pkl
        self.roi_train_val_path = cf.train_val_roi_pkl
        self.roi_test20_path = cf.test20_roi_pkl

        self.data_SYSUCC_external_path = cf.infer_data_pkl  # 院内院外测试
        self.roi_SYSUCC_external_path = cf.infer_roi_pkl  # 院内院外测试

        self.data_root,self.roi_root = self.getData()
        self.data_list = list(self.data_root[self.state.split('&')[0]].keys())
        

    def getData(self):
        '''
        获取label为1的每例数据的路径
        return list形式的路径
        '''
        assert self.state =='train' or self.state == 'val' or self.state =='test' or self.state =='SYSUCC_test' or self.state =='test_ex_anony' or self.state == 'train&foshan' or self.state == 'test_ex_anony&no foshan'
       
        if self.state == 'train':
            roi_root = self.roi_train_val_path
            data_root = self.data_train_val_test_20path

        elif self.state == 'val':
            roi_root = self.roi_train_val_path
            data_root = self.data_train_val_test_20path

        elif self.state == 'test':
            roi_root = self.roi_test20_path
            data_root = self.data_train_val_test_20path

        elif self.state == 'SYSUCC_test' or self.state == 'test_ex_anony':
            roi_root = self.roi_SYSUCC_external_path
            data_root = self.data_SYSUCC_external_path

        # 佛山数据加入训练
        elif self.state == 'train&foshan':
            roi_root = self.roi_train_val_path
            data_root = self.data_train_val_test_20path

            data_pkl = self.get_pkl(data_root)
            roi_pkl = self.get_pkl(roi_root)

            test_ex_anony_roi_root = self.roi_SYSUCC_external_path
            test_ex_anony_data_root = self.data_SYSUCC_external_path

            test_ex_anony_data_pkl = self.get_pkl(test_ex_anony_data_root)
            test_ex_anony_roi_pkl = self.get_pkl(test_ex_anony_roi_root)

            t, hospital = self.state.split('&')[0],self.state.split('&')[1]
            for k in test_ex_anony_data_pkl['test_ex_anony'].keys():
                if hospital in k:
                    data_pkl[t][k]=test_ex_anony_data_pkl['test_ex_anony'][k]
            for j in test_ex_anony_roi_pkl['test_ex_anony'].keys():
                if hospital in j:
                    roi_pkl[j]=test_ex_anony_roi_pkl['test_ex_anony'][j]

            return data_pkl, roi_pkl

        # 外部测试集不测试佛山数据
        elif self.state == 'test_ex_anony&no foshan':
            test_ex_anony_roi_root = self.roi_NPC_test_new
            test_ex_anony_data_root = self.NPC_test_new
            test_ex_anony_data_pkl = self.get_pkl(test_ex_anony_data_root)
            test_ex_anony_roi_pkl = self.get_pkl(test_ex_anony_roi_root)
            tmp = copy.deepcopy(test_ex_anony_data_pkl)  # 防止出栈时改变原来的大小:dictionary changed size during iteration
            for k in tmp['test_ex_anony'].keys():
                if 'foshan' in k:
                    test_ex_anony_data_pkl['test_ex_anony'].pop(k)

            return test_ex_anony_data_pkl, test_ex_anony_roi_pkl


        return self.get_pkl(data_root), self.get_pkl(roi_root)


    def get_pkl(self,root):
        '''
        获取检测的pkl文件
        return pkl对象
        '''
        with open(root, "rb") as f:
            dict_data = pickle.load(f)

            return dict_data


    def normalized(self, img):
        '''
        z-score标准化
        :param ct_array：标准化前的array格式的数据
        return 返回标准化后的array格式的数据
        '''
        lower_bound = np.percentile(img, 0.5)
        upper_bound = np.percentile(img, 99.5)
        mask = (img > lower_bound) & (img < upper_bound)
        img = np.clip(img, lower_bound, upper_bound)# 截取,<lower_bound的=lower_bound
        mn = img[mask].mean()
        std = img[mask].std() + 1e-5
        img = (img - mn) / std

        return img


    def data_enhancement(self,img,mask):
        '''
        function:数据增强
        '''
        # 亮度调节
        if np.random.rand() <= 0.5:
            scale = np.random.uniform(low=1.0 - 0.5, high=1.0 + 0.5)
            img= exposure.adjust_gamma(img, scale) #>1调暗,<1调亮

        # # zoom缩放
        # if np.random.rand() <= 0.5:
        #     scale = np.random.uniform(low=1.0 - 0.5, high=1.0 + 0.5)
        #     img = zoom(img,zoom=[1,scale,scale],order=1)
        #     mask = zoom(mask,zoom=[1,scale,scale],order=0)

        #  randomly scale
        if np.random.rand() <= 0.5:
            scale = np.random.randint(-20, 20)
            img = rotate(img,scale,axes=(2, 1),reshape=False,order=1)
            mask = rotate(mask,scale,axes=(2, 1),reshape=False,order=0)

        if np.random.rand() <= 0.5:
        # randomly flipping
            flip_num = np.random.randint(0, 3)
            if flip_num == 0:
                img = np.flipud(img).copy()  # 上下翻转
                mask = np.flipud(mask).copy()  # 上下翻转
            elif flip_num == 1:
                img = np.fliplr(img).copy()  # 水平翻转
                mask = np.fliplr(mask).copy()  # 水平翻转
            elif flip_num == 2:
                img = np.flipud(img).copy()
                img = np.fliplr(img).copy()
                mask = np.flipud(mask).copy()
                mask = np.fliplr(mask).copy()

        return img, mask


    def __getitem__(self, index):

        name = self.data_list[index]
        pic_path = self.data_root[self.state.split('&')[0]][name]['img_path'].replace('data_local/2021_Data','/data/data_local_to_data/NPC')
        label_path = self.data_root[self.state.split('&')[0]][name]['mask_path'].replace('data_local/2021_Data','/data/data_local_to_data/NPC')

        ct = sitk.ReadImage(pic_path)   # z, y, x
        ct_array = sitk.GetArrayFromImage(ct)
        
        seg = sitk.ReadImage(label_path,sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        origin_shape = seg_array.shape
        
        # crop 根据ROI坐标裁剪
        if self.state in ['train','train&foshan','val','test']:
            zxy = self.roi_root[name if self.state!='test' else (name+'.mha')]
            states = self.state.split('&')
            if len(states)>1 and states[1] in name:
                if zxy[5] - zxy[4] <300:
                    zxy[5] = seg_array.shape[1] - 2

        elif self.state in ['SYSUCC_test','test_ex_anony','test_ex_anony&no foshan']:
            zxy = self.roi_root[self.state.split('&')[0]][name]
            if zxy[5] - zxy[4] <300:
                zxy[5] = seg_array.shape[1] - 2
       
        # 只裁XY方向
        ct_array = ct_array[:,zxy[4]:zxy[5]+1,zxy[2]:zxy[3]+1]
        seg_array = seg_array[:,zxy[4]:zxy[5]+1,zxy[2]:zxy[3]+1]
        crop_shape = seg_array.shape  # 测试时，resize到crop时的大小需要用到
       
        # resize 调整大小
        shape_scale = np.array(self.input_size) / ct_array.shape
        ct_array = zoom(ct_array, shape_scale, order=1)
        seg_array = zoom(seg_array, shape_scale, order=0)

        spacing = seg.GetSpacing()
        origin = seg.GetOrigin()
        direction = seg.GetDirection()

        if self.state == 'train' or self.state == 'train&foshan':
            ct_array, seg_array = self.data_enhancement(ct_array,seg_array)

        
        ct_array = self.normalized(ct_array)
     
        ct_array = ct_array[np.newaxis,:,:,:]
        seg_array = seg_array[np.newaxis,:,:,:]

        ct_array, seg_array = torch.from_numpy(ct_array.astype(np.float32)), torch.from_numpy(seg_array.astype(np.float32))
        
        
        return ct_array, seg_array, name,origin_shape,crop_shape,zxy,spacing,origin,direction


    def __len__(self):
        return len(self.data_list)
