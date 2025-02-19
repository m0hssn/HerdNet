import os
import PIL
import pandas as pd
import numpy as np
from typing import Optional, List, Any, Dict, Union, Iterable

from ..data.types import BoundingBox
from ..data.utils import group_by_image
from .register import DATASETS
from .csv import CSVDataset

@DATASETS.register()
class FolderDataset(CSVDataset):
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
    ) -> None:
        super(FolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)
        
        self.folder_images = [i for i in os.listdir(self.root_dir) if i.lower().endswith(('.jpg', '.jpeg'))]
        self._img_names = self.folder_images
        
        self.anno_keys = self.data.columns
        self.data['from_folder'] = 0
        
        folder_only_images = np.setdiff1d(self.folder_images, self.data['images'].unique().tolist())
        folder_df = pd.DataFrame({'images': folder_only_images, 'from_folder': 1, 'labels':0})
        
        
        self.data = pd.concat([self.data, folder_df], ignore_index=True).convert_dtypes()
        self._ordered_img_names = group_by_image(self.data)['images'].values.tolist()
    
    def _load_image(self, index: int) -> PIL.Image.Image:
        img_name = self._ordered_img_names[index]
        img_path = os.path.join(self.root_dir, img_name)
        pil_img = PIL.Image.open(img_path).convert('RGB')
        pil_img.filename = img_name
        return pil_img
    
    def _load_target(self, index: int) -> Dict[str, List[Any]]:
        img_name = self._ordered_img_names[index]
        annotations = self.data[self.data['images'] == img_name]
        anno_keys = list(self.anno_keys)
        anno_keys.remove('images')
        
        target = {'image_id': [index], 'image_name': [img_name]}
        
        if not annotations[anno_keys].isnull().values.any():
            for key in anno_keys:
                target[key] = list(annotations[key])
                if key == 'annos':
                    target[key] = [list(a.get_tuple) for a in annotations[key]]
        else:
            for key in anno_keys:
                if self.anno_type == 'BoundingBox':
                    target[key] = [[0, 1, 2, 3]] if key == 'annos' else [0]
                else:
                    target[key] = []
        
        return target
    
    def _get_single_item(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)
        return self._transforms(image, target)
    
    def __getitem__(self, index: Union[int, Iterable[int]]):
        if isinstance(index, (list, tuple, np.ndarray)):
            return [self._get_single_item(i) for i in index]
        return self._get_single_item(index)
