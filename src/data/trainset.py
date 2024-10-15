import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class TRAINSETDataset(CDDataset):  
    def __init__(       
        self,     
        root, phase='train',  
        transforms=(None, None, None),  
        repeats= 1,  
        subset='val'    # 
    ):     
        super().__init__(root, phase, transforms, repeats, subset)        

    def _read_file_paths(self):         # 这个方法实际上就是根据你下载的数据集(已经预处理裁剪好的数据集)的目录结构，来获取图片列表的。
    
        self.t1_list = sorted(glob(join(self.root, self.subset, 'T1', '**', '*.tif'), recursive=True))
        self.t2_list = sorted(glob(join(self.root, self.subset, 'T2', '**', '*.tif'), recursive=True))
        self.tar_list = sorted(glob(join(self.root, self.subset, 'gt', '**', '*.tif'), recursive=True))
        assert len(self.t1_list) == len(self.t2_list) == len(self.tar_list)
        return self.t1_list, self.t2_list, self.tar_list       

    # def fetch_target(self, target_path):
    #     label = super().fetch_target(target_path)
    #     if label.dtype == np.float32 or label.dtype == np.float64:
    #         # 如果图像数据是浮点型，将其转换为8位整型
    #         label = label.astype(np.uint8)
    #     classes = {200: 0, 150: 1, 100: 2, 250: 3, 220: 4, 50: 5, 0: 6}
    #     # 将每个像素值转换为对应的类别索引
    #     for pixel_value, class_index in classes.items():
    #         label[label == pixel_value] = class_index
    #     return label
    def fetch_target(self, target_path):
        import cv2
        label  = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        classes = {200: 0, 150: 1, 100: 2, 250: 3, 220: 4, 50: 5, 0: 6}
        converts_labels = np.zeros_like(label, dtype=np.uint8)
        # 将每个像素值转换为对应的类别索引
        for pixel_value, class_index in classes.items():
            converts_labels[label == pixel_value] = class_index

        return converts_labels
