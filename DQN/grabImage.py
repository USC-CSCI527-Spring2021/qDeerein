from PIL import Image
from PIL import ImageGrab, ImageEnhance, ImageFilter, ImageOps 
from grabscreen import grab_screen
from matplotlib import pyplot as plt
import pytesseract
import time
import numpy as np
import cv2
import io
# im1 = Image.open(r"C:\Users\Akash\Pictures\Screenshots\dave8.png") 
# im1.show() 
# time.sleep(3)
# im = ImageGrab.grab(bbox = (310,0,470,47)) 
# im.save('temp.jpg')
# im.show()
# im = im.filter(ImageFilter.MedianFilter())
# enhancer = ImageEnhance.Contrast(im)
# im = enhancer.enhance(2)
# im = im.convert('1')
# im.show()
# img_cv = cv2.imread('temp2.jpg')
# imgGrayScaled = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(imgGrayScaled, cv2.COLOR_BGR2RGB))
# #print(imgGrayScaled)
# imgBlurred = cv2.GaussianBlur(imgGrayScaled, (5, 5), 0)
# #print(imgBlurred)
# grayImage = cv2.bilateralFilter(imgBlurred, 11, 17, 17)
# #print(grayImage)
# #print(new_image)

# configr = ('--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
# print(pytesseract.image_to_string(imgGrayScaled,config=configr))
# text = pytesseract.image_to_string(Image.open('temp2.jpg'), config='-l eng --oem 1 --psm 3')
# print(text)
img = Image.open(r'D:\USC\ML for Games\ErrorLogs\50E20G-02-13-2021-0106\reward.jpg')
# img = cv2.imread(r'D:\USC\ML for Games\ErrorLogs\50E20G-02-13-2021-0106\reward.jpg')
# img = img.convert('L')
# img.save('reward1.jpg')
# thresh = 125
# fn = lambda x : 255 if x > thresh else 0
# img = img.convert('L').point(fn, mode='1')
# r.save('foo.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


width, height = img.size
for x in range(1,18):
    new_size = width*x, height*x
    img = img.resize(new_size,Image.LANCZOS)
    # img = img.convert('L')
    # img = img.point(lambda x: 0 if x < 150 else 255, '1')
    # img.show()
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    # img = cv2.imread('reward.jpg')
    imagetext = pytesseract.image_to_string(img,config='digits')
    print("Value of %d is %s" % (x, imagetext))
    ans=''
    for c in imagetext:
        if c.isdigit():
            ans+=c
    print("----",int(ans),len(ans))
