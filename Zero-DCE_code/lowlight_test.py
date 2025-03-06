import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from pathlib import Path

from torchvision import transforms
from PIL import Image
import glob
import time

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
    image_dir = Path('/home/ubuntu/data/DarkFace_Train_2021/image')
    save_dir = Path('/home/ubuntu/data/DarkFace_Train_2021/Zero-DCE')
    # image_dir = Path('../LOD/RGB_Dark')
    # save_dir = Path('../LOD/RGB_ZeroDCE')
    save_dir.mkdir(exist_ok=True, parents=True)
    images = sorted(image_dir.glob('*'))

    DCE_net = model.enhance_net_nopool().cuda()
    # DCE_net.load_state_dict(torch.load('LOD/Epoch199.pth'))
    DCE_net.load_state_dict(torch.load('DarkFace/Epoch1.pth'))

    with torch.no_grad():
        for image in images:
            print(image)
            save_path = save_dir / image.name
            lowlight(DCE_net, image, save_path)
            # if file_path.is_dir():
            #     test_list = glob.glob(str(file_path) + "/*")
            #     for image in test_list:
            #         print(image)
            #         lowlight(image)
            # else:
            #     print(file_path)
            #     lowlight(str(file_path))
