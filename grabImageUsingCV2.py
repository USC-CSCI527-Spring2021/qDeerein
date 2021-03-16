from PIL import Image
from PIL import ImageGrab, ImageEnhance, ImageFilter, ImageOps 
from grabscreen import grab_screen
from matplotlib import pyplot as plt
import pytesseract
import time
import numpy as np
import cv2
import io

def fetchScoreFromImage(img, x=6):
    # kernel = np.ones((3,3),np.uint8)
    (thresh, img) = cv2.threshold(img[:,:,1], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("Image", img)
    # cv2.imwrite(path+'rewardBW.jpg', img)
    img = cv2.subtract(255, img)
    # cv2.imwrite(path+'rewardWB.jpg', img)
    height, width = img.shape
    # x = 10
    new_size = width*x, height*x
    print(new_size)
    
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # cv2.imwrite(path+'rewards/reward'+ str(x) +'.jpg', img)
    imagetext = pytesseract.image_to_string(img,config='digits')
    # print("\n\nValue of %d is %s" % (x, imagetext))
    # ans=''
    try:
        answer = int(imagetext)
        print("Score fetched ", answer, len(imagetext))
        return answer
    except:
        print("ERROR: Score Fetched ", imagetext)
        return 0