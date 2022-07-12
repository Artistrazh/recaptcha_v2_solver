import os
import re
import json
import argparse
from typing import Union, Dict
from dataclasses import dataclass

import cv2
import yaml
import spacy
import torch
import numpy as np
from loguru import logger
from torchvision import transforms
from skimage.restoration import estimate_sigma
from torchvision.transforms.functional import InterpolationMode

try:    
    from BLIP.models.blip import blip_decoder
    logger.info('Decoder import successfull')
except ModuleNotFoundError:
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from BLIP.models.blip import blip_decoder
    logger.info('Decoder import successfull')
     
from args_parsing import server_parser

try:
    from service_funcs import BaseInit, crop_image, lemming_string, get_img_from_url, get_synonyms
except ModuleNotFoundError:
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from service_funcs import BaseInit, crop_image, lemming_string, get_img_from_url, get_synonyms
    logger.info('service_funcs import successfull blip')
    
with open('/app/pathes.yaml') as f:
    pathes = yaml.safe_load(f)

parser = server_parser()
args = parser.parse_args()

IMAGE_SIZE = 384
device = torch.device('cuda') if torch.cuda.is_available() and args.device.lower().split() != 'cpu' else torch.device('cpu')

logger.info(f'Now device is {device}')
logger.info('Service is starting!')
logger.info('Loading BLIP weights...')

@dataclass
class BlipModels:
    '''Class for initializing BLIP models.'''
    base_model_path: str = pathes['base_model_path']
    large_model_path: str = pathes['large_model_path']
    
    def __post_init__(self):
        try:
            self.model_base = blip_decoder(pretrained=self.base_model_path, vit='base', image_size=IMAGE_SIZE) 
            self.model_large = blip_decoder(pretrained=self.large_model_path, vit='large', image_size=IMAGE_SIZE)
        except RuntimeError as e:
            logger.error('Failed load BLIP model!', e)
        try:
            self.model_base.eval()
            self.model_large.eval()
        except (AttributeError, NameError) as e:
            logger.error('Failed to call eval() method! on BLIP model!', e)
        try:
            self.model_base.to(device)
            self.model_large.to(device)
        except (AttributeError, NameError) as e:
            logger.error('Failed to transfer BLIP model to device!', e)
        
    def __repr__(self):
        return 'BlipModels class'
    
BLIP_MODELS = BlipModels()

# Imagenet values uses to transforms.Normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),   
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) 
    ])
   
    
class BlipMatcher(BaseInit):
    '''A class that implements the logic of captioning pictures 
    and searching for matching ones by description.
    '''
    def __init__(self, target_cat, image, n_squares):
        super().__init__(target_cat, image, n_squares)

    def estimate_noise(self, image: np.ndarray) -> float:
        '''Estimate of the Gaussian noise standard deviation.'''
        return estimate_sigma(image, average_sigmas=True)

    def estimate_images(self, images_dict: Dict[int, Union[str, np.ndarray]]) -> Dict[int, Dict[float, np.ndarray]]:
        '''Estimate noise standard deviation for some images.'''
        mapping = {}
        for i in images_dict:
            if isinstance(images_dict[i], str):
                sigma = self.estimate_noise(get_img_from_url(images_dict[i]))
                mapping[i] = {'sigma': sigma, 'image': get_img_from_url(images_dict[i])}
            else:
                sigma = self.estimate_noise(images_dict[i])
                mapping[i] = {'sigma': sigma, 'image': images_dict[i]}
        return mapping

    def image_denoising(self, images: Dict[int, Union[str, np.ndarray]], filter_force: int=10, threshold:int=8) -> Dict[int, np.ndarray]:
        '''Denoising image if noise standard deviation more than threshold else return image.'''
        print('demoising here')
        denoised_dict = {}
        img_sigma_mapping = self.estimate_images(images)
        for (i, image_dict) in img_sigma_mapping.items():
            if image_dict['sigma'] >= threshold:
                logger.info(f"{i} with sigma {image_dict['sigma']} is denoising")
                mdc_img = cv2.fastNlMeansDenoisingColored(image_dict['image'], None, filter_force, filter_force, 7, 21)
                denoised_dict[i] = mdc_img
            else:
                denoised_dict[i] = image_dict['image']
        return denoised_dict

    def match(self, target: str, index_image: int, image_caption: list) -> Union[int, None]:
        '''Return index_image if image_caption contain target else None.'''
        target_words = lemming_string(target) 
        captions = lemming_string(image_caption)
        for word in target_words.split():
            if re.findall(word, captions):
                return index_image
        return None

    def main(self) -> list:
        '''Function that processing two logic branch: Selection-based image CAPTCHA 
        and  Click-based image CAPTCHA.
        
        Parameters
        ----------
        text
            Target string that we getting from request and want to finding in caption to image.
        link_to_picture
            Sctructure that contain images, may be dict or str. 
            
            If we have Selection-based image CAPTCHA we just click on squares and get result:
            we entered or no. In this case link_to_picture is just link to image that we should downloading
            and sign.
            
            If we have Click-based image CAPTCHA we click on squares, other images appear 
            in place of the selected images and we should again click on squares etc.
            In this case link_to_picture is dict that contain squares index and corresponding images.
        squares
            The number of squares into which we cut the image.
            May be == 1, in this case we don't cut image and considered as a whole.            
        '''

        logger.info('We are in main')
        index_images_list = []
        text = get_synonyms(self.target_cat)
        
        if self.n_squares == 1 and isinstance(self.image, dict):
            denoised_dict = self.image_denoising(self.image, filter_force=11, threshold=7)
            for index_image in denoised_dict:
                image_caption = self.get_predict(denoised_dict[index_image]) 
                logger.info(f'Image caption is {image_caption}, {index_image}\n')
                print('-'*30)
                index_image = self.match(text, index_image, image_caption)
                logger.info(f'Index image from match is {index_image}')
                if index_image:
                    index_images_list.append(index_image)
                    logger.info(f'index_images_list {index_images_list}')
        else:
            images_dict = crop_image(self.image, self.n_squares)  
            denoised_dict = self.image_denoising(images_dict, filter_force=11, threshold=7)
            for index_image in denoised_dict:
                image_caption = self.get_predict(denoised_dict[index_image])
                logger.info(f'Image caption is {image_caption}, {index_image}\n')
                print('-'*30)
                index_image = self.match(text, index_image, image_caption)
                if index_image:
                    index_images_list.append(index_image)
        logger.info(f'json.dumps(index_images_list) is {json.dumps(index_images_list)}')
        return index_images_list
        
    def get_predict(self, sample: np.ndarray) -> Union[list, str]:
        '''Get predict image caption from BLIP few different ways: using Vit base/large 
        and Beam Search/Nucleus Sampling.  
        '''
        text = []
        img = transform(sample).unsqueeze(0).to(device)
        
        with torch.no_grad():
            caption_bs_base= BLIP_MODELS.model_base.generate(img, sample=False, num_beams=7, max_length=16, min_length=5)    # beam search
           # caption_ns_base = BLIP_MODELS.model_base.generate(img, sample=True, max_length=16, min_length=5)                 # nucleus sapling
            
            #caption_bs_large = BLIP_MODELS.model_large.generate(img, sample=False, num_beams=7, max_length=16, min_length=5) # beam search
            caption_ns_large = BLIP_MODELS.model_large.generate(img, sample=True, max_length=16, min_length=5)               # nucleus sapling

        text_bs_base = caption_bs_base[0].capitalize().strip() + '.'
        #text_ns_base = caption_ns_base[0].capitalize().strip() + '.'
        #text_bs_large = caption_bs_large[0].capitalize().strip() + '.'
        text_ns_large = caption_ns_large[0].capitalize().strip() + '.'

        # logger.info(f'\ntext_bs_base is {text_bs_base}\n',
        #             f'text_ns_large is {text_ns_large}\n')
        
        # print(f'\ntext_bs_base is {text_bs_base}\n')
        #print(f'text_ns_base is {text_ns_base}\n')
        #print(f'text_bs_large is {text_bs_large}\n')
        # print(f'text_ns_large is {text_ns_large}\n')
        #text.append(caption_bs_base)
        
        text.extend([text_bs_base, text_ns_large])
        return text
