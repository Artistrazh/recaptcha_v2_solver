import sys
import json
import torch
import argparse
import traceback

from loguru import logger

logger.add('debug.log', format="{time} {level} {message}", level="DEBUG", rotation="10 MB", compression="zip")

try:
    from app.Solver.main import main
    logger.info("solver's main import succesfull.")
except:
    import sys
    sys.path.insert(0, '/app/solver')
    from main import main
    logger.info("solver's main import succesfull.")

from PIL import Image
from flask import Flask, request

from YOLO_pipeline import YoloMatcher
from BLIP_pipeline import BlipMatcher 

app = Flask(__name__)

@logger.catch
@app.route("/send_big_picture", methods=["GET", "POST"])
def send_big_picture():
    try:
        json_data = request.json    
        print(json_data)        

        text = json_data['text']
        url = json_data['link_to_picture']
        squares = json_data['squares']
        
        yolo_result = YoloMatcher(text, url, squares).yolo_matcher()
        blip_result = BlipMatcher(text, url, squares).main()
        
        print('blip_result:', blip_result)
        print('yolo_result:', yolo_result)
        return json.dumps(list(set(blip_result)|set(yolo_result)))
    
    except Exception as e:
        logger.error(e)
        traceback.print_exc(file=sys.stdout)
        return 'ERROR'

@logger.catch
@app.route("/send_small_picture", methods=["GET", "POST"])
def send_small_picture():
    try:
        json_data = request.json
        print(json_data)

        # info = {
        #     "text": text,
        #     "pictures": {"3": "link3", "5": "link5"},
        #     "squares": 1
        # }

        text = json_data['text']
        image_with_position = json_data['pictures']
        squares = json_data['squares']
    
        yolo_result = YoloMatcher(text, image_with_position, squares).yolo_matcher()
        blip_result = BlipMatcher(text, image_with_position, squares).main()
        
        print('blip_result:', blip_result)
        print('yolo_result:', yolo_result)
        return json.dumps(list(set(blip_result)|set(yolo_result)))
    
    except Exception as e:
        logger.error(e)
        traceback.print_exc(file=sys.stdout)
        return 'ERROR'
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11013, debug=False)
