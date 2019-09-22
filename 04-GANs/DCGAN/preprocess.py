import os
import matplotlib.pyplot as plt
import argparse
from scipy.misc import imresize

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input folder")
ap.add_argument("-o", "--output", required=True, help="Path to output folder")
ap.add_argument("-is", "--img_size", type=int, default = 64,help="Image size")
args = vars(ap.parse_args())

# root path depends on your computer
root = args["input"]
save_root = args["output"]
resize_size =args["img_size"]

if not os.path.isdir(save_root):
    os.mkdir(save_root)

img_list = os.listdir(root)

for i in range(len(img_list)):
    img = plt.imread(os.path.join(root,img_list[i]))
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=os.path.join(save_root,img_list[i]), arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)