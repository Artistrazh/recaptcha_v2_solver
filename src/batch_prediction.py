from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import os
import numpy as np
import cv2
try:
    from BLIP.models.blip import blip_decoder
    print('decoder import successfull')
except ModuleNotFoundError:
    import sys
    print(sys.path)
    print('--Trying import blip decoder--')
    sys.path.insert(0, '/src/BLIP')
    print(sys.path)
    from BLIP.models.blip import blip_decoder
    print('decoder import successfull')
device = torch.device('cuda')
IMAGE_SIZE = 384
base_model_path = '/weights/BLIP_weights/model*_base_caption.pth'
model_base = blip_decoder(pretrained=base_model_path, vit='base', image_size=IMAGE_SIZE)
model_base.eval()
model_base.to(device)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

img = os.listdir('/app/src/batch_pred_imgs')
print(img)
img = [os.path.join('/app/src/batch_pred_imgs', i) for i in img]
print(img)
img = [cv2.imread(i) for i in img]
print(img)
img = np.asarray(transform(i).unsqueeze(0).to(device) for i in img)
print(img)
#img = np.asarray(img)



with torch.no_grad():
    caption_bs_base = model_base.generate(img, sample=False, num_beams=7, max_length=16, min_length=5)
text_bs_base = caption_bs_base[0].capitalize().strip() + '.'
print('*'*10)
print(text_bs_base)
print('*'*10)

