<div class="recaptcha" align="center">
<h1>Recaptcha V2 Solver</h1>
<p>🚀 Gracefully face Recaptcha V2 challenge. Google Recaptcha V2 automated solution service.</p>
</div>

<div class="solver" align="center">
	<img src="https://github.com/Artistrazh/recaptcha_v2_solver/blob/main/solver.gif">
	<figcaption>10x speed times</figcaption>
</div>


## Motivation 
Recaptcha V2 Solver is a Google Recaptcha V2 automated solution service. Many examples of Recaptcha V2 can be too difficult for a human to solve a captcha problem, it takes from time to time to complete it until the site makes sure that it is a real person. Our service will shift the responsibility for passing the captcha to artificial intelligence, while you drink coffee and use something more important.

During the study of repeated solutions, many tools were found that solve Recaptcha V2 by sound ([one of them](https://github.com/dessant/buster)). Much to my surprise, during the operation, we did not find a single implementation that would solve Recaptcha V2 from pictures. We were presented with an interesting solution for the implementation of Recaptcha, about getting images.

## How to use 

### With Docker
1. mkdir recaptcha_v2_solver
2. cd recaptcha_v2_solver 
3. git clone https://github.com/Artistrazh/recaptcha_v2_solver
4. make up
5. make go or python3 /app/src/run.py --device {cuda or cpu, default cuda}  
after run.py it's ready to go -> python3 /app/solver/main.py --socks {optional, your socks} --links {optional, links to webpage with Google RecaptchaV2, default test link} 

### Without Docker
1. mkdir recaptcha_v2_solver 
2. cd recaptcha_v2_solver 
3. git clone https://github.com/Artistrazh/recaptcha_v2_solver
4. pip install -r requirements.text
5. python3 /app/src/run.py --device {cuda or cpu, default cuda}  
after run.py it's ready to go -> python3 /app/solver/main.py --socks {optional, your socks} --links {optional, links to webpage with Google RecaptchaV2, default test link} 

## What's under the hood?
*   Modern BLIP language model for text image subscription - [Article](https://arxiv.org/abs/2201.12086), [GitHub](https://github.com/salesforce/BLIP);
*   YOLOv3 (or optionally YOLOv5) for object detection - [YOLOv3 article](https://arxiv.org/abs/1804.02767), [GitHub YOLOv3](https://github.com/ultralytics/yolov3), [GitHub YOLOv5](https://github.com/ultralytics/yolov5);
*   YOLOv3 trained on a manually assembled and labeled dataset. This model detects categories that are missing in the YOLO implementation trained on MS COCO. In addition, it detects categories where BLIP does not perform well. This [dataset](https://github.com/brian-the-dev/recaptcha-dataset) was taken as the basis for training my model. The model was trained on just over 512 labeled images in three categories: crosswalk, chimney, and stairs. Some of the images were collected from Google Images and also marked up.
The weights of the trained model and the labeled dataset will be posted.

## Results 
While testing our solution we got the best result in 32% of solved captchas while usually the result fluctuated around 23 - 27%. It may seem that this isn't good enough but while I was writing this text I tried to complete the captcha on my own. It took me quite a long time to solve the captcha despite the fact that i'm a person and I always click on the correct squares.

A complete rejection of the BLIP model and the transition to YOLO will reduce the captcha completion time and increase the quality, but this requires marking up a large dataset.

## Limitations
The results may depend on:
*   The number of requests sent from your IP address; 
*   What WebDriver setting you are using;
*   On which web page you are trying to solve the Recaptcha;
*   Whether you use socks to proxy traffic; 
*   And also on the adversial noise that Google probably imposes on Recaptcha images.

Synonyms for blip. We tried to get good synonyms with the Glove models and NLTK
but it didn't work good so we had to hardcode it.

## Memory Usage
Our solution requires about 4.5 GB GPU RAM for normal operation.

## Reference
While searching for existing solutions, [article](https://arxiv.org/pdf/2104.03366.pdf) was found on the subject of our work. This work allowed us to better understand how Google Recaptcha V2 works and to take into account some points when working on our project.
