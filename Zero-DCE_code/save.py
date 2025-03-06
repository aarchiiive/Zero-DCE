import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import time
from tqdm import tqdm

import model

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def lowlight(DCE_net, image_path, save_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    torchvision.utils.save_image(enhanced_image, save_path)

if __name__ == '__main__':
    image_dir = Path('../DarkFace_Train_2021/image')
    save_dir = Path('../DarkFace_ZeroDCE')
    # image_dir = Path('../LOD/RGB_Dark')
    # save_dir = Path('../LOD/RGB_ZeroDCE++')
    # image_dir = Path('../Exdark/JPEGImages/IMGS_dark')
    # save_dir = Path('../Exdark/JPEGImages/IMGS_ZeroDCE++')
    # save_dir = Path('../Exdark_ZeroDCE++')
    save_dir.mkdir(exist_ok=True, parents=True)
    # video_dir = Path('../videos/Exdark')
    video_dir = Path('../videos/DarkFace')
    video_dir.mkdir(exist_ok=True, parents=True)
    images = sorted(image_dir.glob('*'))

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.eval()

    for image_index in tqdm(range(len(images))):
        image = images[image_index]
        results = []
        for i in range(200):
            # DCE_net.load_state_dict(torch.load(f'Exdark/Epoch{i}.pth'))
            DCE_net.load_state_dict(torch.load(f'DarkFace/Epoch{i}.pth'))
            with torch.no_grad():
                save_path = save_dir / f'{i}_{image.name}'
                lowlight(DCE_net, image, save_path)
                results.append(save_path)

        img = cv2.imread(save_path)
        h, w = img.shape[:2]

        # video = cv2.VideoWriter(f'{image.stem}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
        video = cv2.VideoWriter(video_dir / f'{image.stem}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (w * 2, h))

        for i, result in enumerate(results):
            img = cv2.imread(str(image))
            results_img = cv2.imread(str(result))
            # print(img.shape, results_img.shape)
            img = np.hstack((img, results_img))
            # print(img.shape)
            cv2.putText(img, f'Epoch: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            video.write(img)

        video.release()