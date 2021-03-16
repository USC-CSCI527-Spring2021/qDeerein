from PIL import Image
from PIL import ImageGrab, ImageEnhance, ImageFilter, ImageOps 
from grabscreen import grab_screen
from matplotlib import pyplot as plt
import pytesseract
import time
import numpy as np
import cv2
import io

# path = 'D:/USC/ML for Games/ErrorLogs/100E50G-03-13-2021-1025/'
# path = 'D:/USC/ML for Games/ErrorLogs/50E20G-02-13-2021-0102/'
# path = 'D:/USC/ML for Games/ErrorLogs/50E20G-02-13-2021-0106/'
# path = 'D:/USC/ML for Games/ErrorLogs/02-13-2021-0556/'
path = "./"
# img = Image.open(r'D:\USC\ML for Games\ErrorLogs\100E50G-03-13-2021-1025\reward.jpg')
img = cv2.imread(path+'reward.jpg')
kernel = np.ones((3,3),np.uint8)
# cv2.imwrite(path+"B.jpg", img[:,:,0])
# cv2.imwrite(path+"G.jpg", img[:,:,1])
# cv2.imwrite(path+"R.jpg", img[:,:,2])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite(path+'rewardGRAY.jpg', img)
(thresh, img) = cv2.threshold(img[:,:,1], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Image", img)
cv2.imwrite(path+'rewardBW.jpg', img)
# img = cv2.dilate(img,kernel,iterations = 1)
# cv2.imwrite(path+'dilation.jpg', img)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite(path+'closing.jpg', img)
# img = cv2.erode(img,kernel,iterations = 1)
# cv2.imwrite(path+'erode.jpg', img)
img = cv2.subtract(255, img)
cv2.imwrite(path+'rewardWB.jpg', img)
height, width = img.shape

for x in range(1,18):
    new_size = width*x, height*x
    print(new_size)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(path+'rewards/reward'+ str(x) +'.jpg', img)
    # img = img.convert('L')
    # img = img.point(lambda x: 0 if x < 150 else 255, '1')
    # img.show()
    # custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    # img = cv2.imread('reward.jpg')
    imagetext = pytesseract.image_to_string(img,config='digits')
    print("\n\nValue of %d is %s" % (x, imagetext))
    ans=''
    for c in imagetext:
        if c.isdigit():
            ans+=c
    try:
        print(int(ans),len(ans))
    except:
        print(imagetext)