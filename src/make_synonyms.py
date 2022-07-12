''' 
Сreating a dictionary of synonyms for selected words.

The script bypasses a dictionary of ~ 1.5M words and compiles 
a dict of synonyms by cosine distance. With a word vector dimension of 25, 
it takes about 2 minute for each word.

'''

import re
import json
import time
from typing import Union, Dict

import nltk
import yaml
import numpy as np
from tqdm import tqdm
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from scipy.spatial.distance import cosine

with open('/app/pathes.yaml') as f:
    pathes = yaml.safe_load(f)

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

stops = set(stopwords.words('english'))

CATEGORIES = ['bicycle', 'bridge', 'bus', 'car', 
              'chimney', 'crosswalk', 'hydrant', 'motorcycle', 
              'mountain', 'palm']

CAR_SYNONYMS = ['car', 'auto', 'vehicle', 'automobile', 'machine', 'motorcar', 'honda']
N_SYN = 6


def make_emmbed_dict(model_path: str, size: int=200) -> Dict[str, np.ndarray]:
    '''You can choice glove models with 25, 50, 100 and 200 embedding size.'''
    emmbed_dict = {} 
    logger.info('Loading model...')
    try:
        with open(model_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:],'float16')
                emmbed_dict[word]=vector
        logger.info('Successful loading model')
    except (FileExistsError, FileNotFoundError) as e:
        logger.error(e)

    invalid_key = [i for i in emmbed_dict if emmbed_dict[i].shape != (size, )]

    for i in invalid_key:
        emmbed_dict.pop(i)

    return emmbed_dict


def preprocessing(emmbed_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''Preparing dict with words.'''
    key_list, val_list = [], []

    for key, val in emmbed_dict.items():
        key_list.append(re.sub(r'[^\w\s]','',key))
        val_list.append(val)

    new_dict = dict(zip(key_list, val_list))
    new_dict= {i:j for i,j in new_dict.items() if len(i) > 1 and i not in stops}

    return new_dict


def make_synonyms_dict(cat_list: list, clear_dict: Dict[str, np.ndarray]) -> Union[Dict[str, Dict[str, float]], list]:
    '''Making synonyms dict.'''
    cosine_dist_dict, list_words = {}, []

    for j in tqdm(cat_list):
        j = j.lower().strip()
        try:
            if j == 'car': # According to the word 'car' inadequate synonyms, we will set manually.
                cosine_dist_dict[j] = CAR_SYNONYMS
                continue

            cosine_dist_dict[j] = {i:cosine(clear_dict[j], clear_dict[i]) for i in clear_dict}
            cosine_dist_dict[j] = dict(sorted(cosine_dist_dict[j].items(), key=lambda item: item[1]))
            cosine_dist_dict[j] = dict(list(cosine_dist_dict[j].items())[:N_SYN])
            list_words.append(cosine_dist_dict[j].keys())

        except Exception as e:
            logger.warning(e)
            continue

    return cosine_dist_dict, list_words


def transform_dict(cosine_dist_dict: Dict[str, Dict[str, float]]) -> Dict[str, list]:
    '''Making transform.'''
    json_data = {}

    for i in cosine_dist_dict:
        json_data[i] = cosine_dist_dict[i]
    return json_data


def lemming(lst: list) -> set:
    '''Making lemming.'''
    lst = [lemmatizer.lemmatize(i) for i in lst]
    return set(lst)


def lemming_synonyms_dict(json_data: Dict[str, list]) -> Dict[str, list]:
    lem_json_data = {}
    for key, val in json_data.items():
        lem_json_data[key] = list(lemming(val))
    return lem_json_data


def main():
    emmbed_dict = make_emmbed_dict(pathes['glove_25'], size=25)
    clear_dict = preprocessing(emmbed_dict)
    cosine_dist_dict, list_words = make_synonyms_dict(CATEGORIES, clear_dict)
    json_data = transform_dict(cosine_dist_dict)
    lem_json_data = lemming_synonyms_dict(json_data)

    with open('glove_synonyms.json', 'w') as f:
        json.dump(lem_json_data, f)


if __name__ == "__main__":
    start = time.time()
    main()
    logger.info(f'Time to making synonyms is spent {round((time.time()-start)/60, 1)} minute')









