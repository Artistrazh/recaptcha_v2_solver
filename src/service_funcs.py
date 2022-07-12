import re
import json
import math

import requests
import cv2
import spacy
import nltk
import yaml
import numpy as np
from nltk.corpus import wordnet, stopwords
from loguru import logger
from typing import Dict

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_md")
STOPS = set(stopwords.words('english'))

with open('/app/pathes.yaml') as f:
    pathes = yaml.safe_load(f)

class BaseInit:
    def __init__(self, target_cat, image, n_squares):
        self.target_cat = target_cat
        self.image = image
        self.n_squares = n_squares
    
    def __repr__(self):
        return 'Base initializer for BlipMatcher and YoloMatcher classes.'


def get_img_from_url(img_url: str) -> np.ndarray:
    '''Download image from URL.'''
    response = requests.get(img_url)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def crop_image(image_path: str, n_squares: int) -> Dict[int, np.ndarray]:
    '''Cropping image into  squares.'''
    if 'http' in image_path:
        img = get_img_from_url(image_path)
    else:
        img = cv2.imread(image_path)

    height, width, channels = img.shape
    crop_w_size = crop_h_size = int(math.sqrt(n_squares))

    data = {}
    _ = 1
    for ih in range(crop_h_size):
        for iw in range(crop_w_size):
            x = width // crop_w_size * iw
            y = height // crop_h_size * ih
            h = (height // crop_h_size)
            w = (width // crop_w_size)
            current_crop = img[y:y+h, x:x+w]
            data[_] = current_crop
            _ += 1
    return data


def lemming_string(image_caption: str) -> str:
    '''Lemming and removing punctuations.'''
    s = re.sub(r'[^\w\s]', '', str(image_caption)).lower().strip()
    s = ' '.join([word.lemma_ for word in nlp(s) if word.lemma_ not in STOPS])
    return s


def get_key(d: Dict[int, str], val: str) -> int:
    '''Getting category number for the target.'''
    for (k, v) in d.items():
        if re.findall(val, v):
            return k


def cat2indxs(mapping: Dict[int, str], target_cat: str) -> list:
    '''Getting the category number.'''
    indxs = []
    for word in target_cat.split():
        indx = get_key(mapping, word)
        if indx not in indxs:
            indxs.append(indx)
    return indxs
    
    
def get_synonyms(target: str) -> str:
    '''Get synonyms from file if word exist in file else get from wordnet.synset.'''
    synonyms_dict = {}
    try:
        with open(pathes['synonyms_file_path']) as f:
            synonyms_dict = json.load(f)
            logger.info('Synonyms file found succesfull')
    except FileNotFoundError as e:
        logger.warning('File with synonyms not found!', e)
    
    target = ' '.join([i for i in target.split() if i not in STOPS])
    syn = list()
    
    if len(synonyms_dict) != 0:
        for target_word in lemming_string(target).split():
            for key in synonyms_dict:
                if target_word in lemming_string(key):
                    syn.extend(synonyms_dict[key])
                    
    if len(syn) == 0:
        for word in lemming_string(target).split():
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    syn.append(lemma.name()), syn.append(word)
    syn = ' '.join(list(set([i.lower() for i in syn if '_' not in i])))
    logger.info(f'Synonyms for {target} - {syn}')
    return syn
