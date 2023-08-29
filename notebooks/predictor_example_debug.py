# In order to debug, copy the code from jupyter notebook(predictor_example.ipynb) to here on 2023.06.13.
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

image = cv2.imread('/LOG/realman/LLM/segment-anything/notebooks/images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

import sys
import pdb
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/LOG/realman/LLM/segment-anything/weights/sam_vit_h_4b8939.pth"
# sam_checkpoint = "/LOG/realman/LLM/segment-anything/weights/sam_vit_b_01ec64.pth"
# sam_checkpoint = "/LOG/realman/LLM/segment-anything/weights/sam_vit_l_0b3195.pth"
model_type = "vit_h"

device = "cuda"
# # pdb.set_trace()
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# # 统计模型参数量
# num_params = sum(p.numel() for p in sam.parameters())
# print("Number of parameters: {}".format(num_params))

sam.to(device=device)
# # pdb.set_trace()
# # 创建了封装了推理模块的预测类，完成初始化

# # 经image encoder生成图像的embedding
# # ----------------------------------
predictor = SamPredictor(sam)

# import time
# time_start = time.time()
predictor.set_image(image)
# time_end = time.time()
# total_time = time_end - time_start
# print(f"3.the total time of image embeddings is {total_time} seconds!")
# # type(predictor.set_image(image))
# # -----------------------------------------------------------------------

# input_point = np.array([[320, 375]])
# input_label = np.array([1])

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# print(masks.shape)


# # -------------------------------------------------------------
# # Batched prompt inputs 
# input_boxes = torch.tensor([
#     [75, 275, 1725, 850],
#     [425, 600, 700, 875],
#     [1375, 550, 1650, 800],
#     [1240, 675, 1400, 750],
# ], device=predictor.device)

# transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
# masks, _, _ = predictor.predict_torch(
#     point_coords=None,
#     point_labels=None,
#     boxes=transformed_boxes,
#     multimask_output=False,
# )

# masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W

# --------------------------------------------------------------------
# end to end batched inference
image1 = image  # truck.jpg from above
image1_boxes = torch.tensor([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
], device=sam.device)

image2 = cv2.imread('/LOG/realman/LLM/segment-anything/notebooks/images/groceries.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2_boxes = torch.tensor([
    [450, 170, 520, 350],
    [350, 190, 450, 350],
    [500, 170, 580, 350],
    [580, 170, 640, 350],
], device=sam.device)

from segment_anything.utils.transforms import ResizeLongestSide
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

batched_input = [
     {
         'image': prepare_image(image1, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
         'original_size': image1.shape[:2]
     },
     {
         'image': prepare_image(image2, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
         'original_size': image2.shape[:2]
     }
]

batched_output = sam(batched_input, multimask_output=False)

# 主要耗时还是在image encoder环节，基本单张图片处理时间稍长于在1.5s