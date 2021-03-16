from PIL import Image
from PIL import ImageGrab, ImageEnhance, ImageFilter, ImageOps 
from grabscreen import grab_screen
from matplotlib import pyplot as plt
import pytesseract
import time
import numpy as np
import cv2
import io
im1 = Image.open(r"C:\Users\Akash\Pictures\Screenshots\dave10.png") 
im1.show() 
time.sleep(3)
im = ImageGrab.grab(bbox = (413,710,935,768))
width, height = im.size
new_size = width*10, height*10
im = im.resize(new_size, Image.LANCZOS)
im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2GRAY)
imagetext = pytesseract.image_to_string(im)
return "DOOR" in imagetext
