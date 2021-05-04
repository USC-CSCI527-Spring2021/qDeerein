import os
import subprocess
from grabImageUsingCV2 import fetchScoreFromImage
import cv2

img = cv2.imread('reward.jpg')
print(fetchScoreFromImage(img,6))

