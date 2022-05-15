##########################
## Ruifeng Hu
## 05-11-2022
## Lexington, MA
## hurufieng.cn@hotmail.com
##########################

##--------------------------------------------
# Using the Genetic Algorithm(GA) to draw
#   the FireFox logo
#
# Using 100 triangles to draw a FireFox logo,
#   each triangle is a chromosome,
#   and each chromosome has 10 genes:
#       ax,ay, bx,by, cx,cy, r,g,b,a
#
# https://www.pianshen.com/article/7818211114/
##------------------------------------------
import os

import matplotlib
import  matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from triangle import Triangle

target_image = "data/firefox_768.png"
results_folder = "results/firefox_768"
if not os.path.exists(results_folder):
    os.makedirs(results_folder,exist_ok=True)
img = Image.open(target_image).resize((256, 256)).convert('RGBA')
img_size = img.size
target_pixels = [np.array(x) for x in list(img.split())]

t1 = Triangle()
t2 = Triangle()

t1.ax,t1.ay = (100,100)
t1.bx,t1.by = (100,200)
t1.cx,t1.cy = (200,200)
t1.color.r,t1.color.g,t1.color.b = (255,0,0)

t1_img = t1.draw_it()
t1_img.show()

t2.ax,t2.ay = (200,100)
t2.bx,t2.by = (100,200)
t2.cx,t2.cy = (200,200)
t2.color.r,t2.color.g,t2.color.b = (0,255,0)

t2_img = t2.draw_it()
t2_img.show()


best_img = Image.new('RGBA', img_size)
best_img = Image.alpha_composite(best_img, t1_img)
best_img = Image.alpha_composite(best_img, t2_img)
best_img.show()

best_img.save("test.png")














