import os
import re
import sys
import math
from typing import Union
from dataclasses import dataclass

import cv2
import yaml
import json
import torch

try:
    import spacy
except TypeError as e:
    pass
import numpy as np
from loguru import logger
from typing import Dict

try:
    from service_funcs import BaseInit, get_img_from_url, lemming_string, cat2indxs
except ModuleNotFoundError:
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from service_funcs import BaseInit, get_img_from_url, lemming_string, cat2indxs
    logger.info('service_funcs import successfull yolo')
    
with open('/app/pathes.yaml') as f:
    pathes = yaml.safe_load(f)


@dataclass
class YoloModels:
    '''Loading custom trained and default Yolo models.'''
    yolo_path: str = pathes['yolov3_path']
    custom_weights: str = pathes['custom_yolo_weights']
    
    def __post_init__(self):
        try:
            self.my_model = torch.hub.load(self.yolo_path, 'custom', path=self.custom_weights, source='local') 
            self.def_model = torch.hub.load('ultralytics/yolov3', 'yolov3')
        except Exception as e:
            print(e)
            self.my_model = torch.hub.load(self.yolo_path, 'custom', path=self.custom_weights, source='local', force_reload=True) 
            self.def_model = torch.hub.load('ultralytics/yolov3', 'yolov3', force_reload=True)
            
            
    def __repr__(self):
        return 'YoloModels class'

YOLO_MODELS = YoloModels()
logger.info('YOLO models downloading successfull')


def make_mapping() -> Dict[int, str]:
    '''Get the match between category and category number.'''
    my_model = YOLO_MODELS.my_model
    def_model = YOLO_MODELS.def_model

    my_cat = my_model.__dict__['names']
    def_cat = def_model.__dict__['names']

    my_cat_mapping = dict(zip(range(len(my_cat)), my_cat))
    def_cat_mapping = dict(zip(range(len(def_cat)), def_cat))
    return my_cat_mapping, def_cat_mapping

MY_CAT_MAPPING, DEF_CAT_MAPPING = make_mapping()


class YoloMatcher(BaseInit):   
    '''A class that implements the logic of object detection 
    and searching for matching ones.
    ''' 
    MY_CAT = ['stair', 'chimney', 'crosswalk', 'cross walk'] # Сategories in which custom yolo was trained

    def __init__(self, target_cat: str, image: Union[str, np.ndarray, dict], n_squares: int):
        super().__init__(target_cat, image, n_squares)
        self.model, self.mapping  = self.choicing_model_and_mapping(self.target_cat)
        self.indxs = cat2indxs(self.mapping, lemming_string(self.target_cat))    

    def xywhn2xyxy(self, coords: list, W: int, H: int) -> list:
        '''Transfrom coord format from xywhn to xyxy.

        Parameters
        ----------
        coords
            Contain x, y, w, h coords, where x and y center coordinates 
            and w, h is width and height respectively. 
        W
            The width of the original image relative to which the 
            transformation takes place.
        H
            The height of the original image relative to which the
            transformation takes place.
        '''
        def replace_neg(l):
            '''Replace all negative coordinates to zero.'''
            indx_neg = [l.index(i) for i in l if i < 0]
            for i in range(len(l)):
                if i in indx_neg:
                    l[i] = 0
            return l 
        x_true = coords[0] * W
        y_true = coords[1] * H
        w_true = coords[2] * W
        h_true = coords[3] * H

        x_min = x_true - (w_true / 2)
        y_min = y_true - (h_true / 2)
        x_max = x_true + (w_true / 2)
        y_max = y_true + (h_true / 2)

        l = [x_min, y_min, x_max, y_max]
        l = replace_neg(l)
        return l

    def is_cross(self, bbox_coord: list, im_coords: list) -> bool:
        '''Check if Bounding Box is crossing image.'''
        ax1,ay1,ax2,ay2 = bbox_coord
        bx1, by1, bx2, by2 = im_coords

        xA = [ax1,ax2]  
        xB = [bx1,bx2]  
        yA = [ay1, ay2] 
        yB = [by1, by2] 

        if max(xA) < min(xB) or min(yA) > max(yB) or min(xA) > max(xB) or max(yA) < min(yB):
            return False
        else:
            # Intersection
            return True
    
    def change_bboxes_coord(self, bbox_coord: list, delta: float=0.05, mode: str='dec') -> list:
        '''Change the size of the bounding box: reduce or enlarge.'''
        assert len(bbox_coord) == 4, 'Size of bbox_coord must be 4 -> (xcenter, ycenter, w, h).'
        assert mode in ['dec', 'inc'], 'Mode most be "dec" or "inc".'
        
        if mode == 'dec':
            new_w = bbox_coord[2] * (1 - delta)
            nen_h = bbox_coord[3] * (1 - delta)
        elif mode == 'inc':
            new_w= bbox_coord[2] * (1 + delta)
            nen_h = bbox_coord[3] * (1 + delta)
        return [*bbox_coord[:2], new_w, nen_h]
    
    def choicing_model_and_mapping(self, target_cat: str):
        try:
            with open(pathes['synonyms_file_path']) as f:
                synonyms_dict = json.load(f)
                logger.info('Synonyms file found succesfull')
        except FileNotFoundError as e:
            logger.warning('File with synonyms not found!', e)
            
        model = None
        for cat in self.MY_CAT:
            if lemming_string(cat) in synonyms_dict:
                cats = synonyms_dict[lemming_string(cat)]
            for cat in cats:
                if re.findall(cat, target_cat):
                    logger.info('Loading custom YOLO model')
                    model = YOLO_MODELS.my_model
                    mapping = MY_CAT_MAPPING
                    break
                else:
                    continue

        if model is None:
            logger.info('Loading default YOLO model')
            model = YOLO_MODELS.def_model
            mapping = DEF_CAT_MAPPING
        return model, mapping 
            
    def get_predict_drom_dict(self, images: Dict[int, str]) -> list:
        output_from_dict = []
        for (number, links) in images.items():
            img = get_img_from_url(links)
            images[number] = img

        for (number, img_arr) in images.items():
            res = self.model(img_arr)
            table_data = res.pandas().xywhn[0]
            class_column = int(list(table_data.columns).index('class'))  

            for row in range(len(table_data)):
                for indx in self.indxs:
                    if indx == table_data.iloc[row, class_column] and number not in output_from_dict:
                        output_from_dict.append(number)
        return output_from_dict 
    
    def yolo_matcher(self) -> list: 
        ''''Get number of squares containing images with target_cat.'''
        target_cat = lemming_string(self.target_cat)
                
        if isinstance(self.image, np.ndarray):
            img = self.iamge
        elif isinstance(self.image, str):
            if 'http' in self.image:
                img = get_img_from_url(self.image)
            else:
                img = cv2.imread(self.image) 
        elif isinstance(self.image, dict) and self.n_squares == 1:
            return self.get_predict_drom_dict(self.image)
        
        if not isinstance(self.image, dict) and self.n_squares != 1:
            height, width = img.shape[0], img.shape[1]
            crop_w_size = crop_h_size = int(math.sqrt(self.n_squares))

            indx = [int(i) for i in range(1, self.n_squares+1)]
            data = []
            for ih in range(crop_h_size):
                for iw in range(crop_w_size):
                    x_min = width // crop_w_size * iw
                    y_min = height // crop_h_size * ih

                    h = (height // crop_h_size)
                    w = (width // crop_w_size)
                    current_crop = img[y_min:y_min+h, x_min:x_min+w]

                    coord = [x_min, y_min, x_min + width // crop_w_size, y_min + height // crop_h_size]
                    data.append((current_crop, coord))
            data = list(zip(indx, data))

            result_all = self.model(img)
            table_data_all = result_all.pandas().xywhn[0]

            class_column = int(list(table_data_all.columns).index('class'))   
            target_bboxes_crop = []
            target_bboxes_all = []

            for row in range(len(table_data_all)):
                for indx in self.indxs:
                    if indx == table_data_all.iat[row, class_column]:
                        target_bboxes_all.append(table_data_all.iloc[row, :4])

            # bboxes with target_cat      
            for i in range(len(data)):
                result_crop = self.model(data[i][1][0])
                table_data_crop = result_crop.pandas().xywhn[0]        
                for row in range(len(table_data_crop)):
                    for indx in self.indxs:
                        if indx == table_data_crop.iat[row, class_column]:
                            target_bboxes_crop.append(table_data_crop.iloc[row, :4])

            #cropping matching
            output_crop = []
            output_all = []
            for i in range(len(target_bboxes_crop)):
                x, y, w, h = target_bboxes_crop[i][:4]
                for j in range(len(data)):
                    img_crop_coord = data[j][1][1]
                    if (self.is_cross(self.xywhn2xyxy(self.change_bboxes_coord([x, y, w, h], delta=0.1), 
                    width // crop_w_size, height // crop_h_size), img_crop_coord) 
                    and data[j][0] not in output_crop):
                        output_crop.append(data[j][0])

            #full matching
            for i in range(len(target_bboxes_all)):
                x, y, w, h = target_bboxes_all[i][:4]
                for j in range(len(data)):
                    img_crop_coord = data[j][1][1]
                    if (self.is_cross(self.xywhn2xyxy(self.change_bboxes_coord([x, y, w, h], delta=0.1), 
                    width, height), img_crop_coord) 
                    and data[j][0] not in output_all):
                        output_all.append(data[j][0])                             
            return list(set(output_crop)|set(output_all)) 


