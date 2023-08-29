# In order to debug, copy the code from jupyter notebook(automatic_mask_geneeator_example.ipynb) to here on 2023.06.14.
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image = cv2.imread('/LOG/realman/LLM/segment-anything/notebooks/images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import sys
import time
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/LOG/realman/LLM/segment-anything/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

sam.to(device=device)

# print(sam)
# image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
# image_tensor = image_tensor.unsqueeze(dim=0).to(device=device)
# time1 = time.time()
# a = sam.image_encoder(image_tensor)
# time2 = time.time()
# print(time2-time1)
mask_generator = SamAutomaticMaskGenerator(sam)

import time
time_start = time.time()
masks = mask_generator.generate(image)
time_end = time.time()
total_time = time_end - time_start
print(f"The total time of mask_generator of image is {total_time} seconds!")