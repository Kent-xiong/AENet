import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class CLCDDataset(CDDataset):  
    def __init__(       
        self,     
        root, phase='train',  
        transforms=(None, None, None),  
        repeats= 1,  
        subset='val'    # 
    ):     
        super().__init__(root, phase, transforms, repeats, subset)        

    def _read_file_paths(self):         # 这个方法实际上就是根据你下载的数据集(已经预处理裁剪好的数据集)的目录结构，来获取图片列表的。
        t1_list = sorted(glob(join(self.root, self.subset, 'time1', '*.png')))     # glob()匹配指定路径模式的文件， 必须要sort排序，图像才能一一对应，因为你预处理好的图像可能是无序的
        t2_list = sorted(glob(join(self.root, self.subset, 'time2', '*.png')))               
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '*.png')))        
        assert len(t1_list) == len(t2_list) == len(tar_list)     
        return t1_list, t2_list, tar_list       

    def fetch_target(self, target_path):            #  对于标签的图像，黑色区域为0 ，白色区域为255。
        return (super().fetch_target(target_path)/255).astype(np.bool)    