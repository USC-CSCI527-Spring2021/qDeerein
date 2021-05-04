from imutils.perspective import four_point_transform
from imutils import contours
from time import sleep
import numpy as np
import imutils
import cv2
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

path = 'D:/USC/ML for Games/ErrorLogs/100E50G-03-13-2021-1025/'
# path = 'D:/USC/ML for Games/ErrorLogs/50E20G-02-13-2021-0102/'
# path = 'D:/USC/ML for Games/ErrorLogs/50E20G-02-13-2021-0106/'
# img = Image.open(r'D:\USC\ML for Games\ErrorLogs\100E50G-03-13-2021-1025\reward.jpg')
img = cv2.imread(path+'reward.jpg')

(thresh, img) = cv2.threshold(img[:,:,1], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow("Image", img)
cv2.imwrite(path+'rewardWB.jpg', img)
# img = cv2.subtract(255, img)
# cv2.imwrite(path+'rewardBW.jpg', img)
# thresh = cv2.threshold(img, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
height, width = img.shape

# for x in range(1,18):
#     new_size = width*x, height*x
    
#     img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
#     cv2.imwrite(path+'rewards/reward'+ str(x) +'.jpg', img)

contours, heirarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

# img_contours = np.zeros(img.shape)
# img_contours = np.zeros(img.shape[:2], dtype="uint8") * 255
# for contour in contours:
    # x, y, width, height = cv2.boundingRect(contours)
    # roi = img[y:y+height, x:x+width]
    # cv2.imwrite(path+"results/After"+str(x)+".jpg", roi)
# cv2.drawContours(img_contours, contours, -1, 0, -1)
# imgs = cv2.bitwise_and(img, img, mask=img_contours)
# # cv2.imwrite(path+"results/Mask.jpg", img_contours)
# cv2.imwrite(path+"results/After"+str(x)+".jpg", imgs)


# draw the contours on the empty image
# cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
#save image
# cv2.imwrite(path+'results/contours.png',img_contours) 
# cv2.imwrite(path+"results/Boundries.jpg", thresh)
# sleep(10)

cnts = imutils.grab_contours((contours, heirarchy))
digitCnts = []
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # if the contour is sufficiently large, it must be a digit
    # if w >= 15 and (h >= 30 and h <= 40):
    digitCnts.append(c)
    roi = img[y:y+height, x:x+width]
    cv2.imwrite(path+"results/After"+str(x)+".jpg", roi)

# sort the contours from left-to-right, then initialize the
# actual digits themselves
# print(digitCnts)
# digitCnts = contours.sort_contours(digitCnts,
# 	method="left-to-right")[0]
# digits = []