from utils.helpers import extract_image_paths
import cv2
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import shutil 
from skimage.exposure import equalize_adapthist

# copy the folder containing all images

src_folder = 'data/microscope_images'
dst_folder = 'data/microscope_images_edge_detection'

#shutil.copytree(src_folder, dst_folder)

# save all paths

paths10=extract_image_paths(dst_folder, image_type='all', magnification=10)
paths40=extract_image_paths(dst_folder, image_type='all', magnification=40)

paths=paths10+paths40

for path in paths:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # cv2 loads images in BGR format by default
    resized_image = cv2.resize(gray_image, (512, 384), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(resized_image, 50, 75)

    # # Visualize the original and edge-detected images side by side
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # ax[0].imshow(resized_image, cmap='gray')
    # ax[0].axis('off')

    # ax[1].imshow(edges, cmap='gray')
    # ax[1].axis('off')

    # plt.show()
    # break

    transform = transforms.Compose([
    transforms.ToTensor()])

    tensor = transform(edges)

    new_path = os.path.splitext(path)[0] + '.pt'

    # save in new folder
    torch.save(tensor, new_path)

    #delete old image
    os.remove(path)