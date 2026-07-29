import cv2
import numpy as np
import matplotlib.pyplot as plt

dir = r'F:\CIME_MS\adnane\20260610_121314'
img_1 = cv2.imread(dir + r'\color_75649.png')
img_2 = cv2.imread(dir + r'\color_75650.png')
roi = img_1[350:, 200:600]
R = roi[:, :, 2].astype(float)
G = roi[:, :, 1].astype(float)
B = roi[:, :, 0].astype(float)
hsv_1 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

score = np.mean(R / (G + B + 1))

roi = img_2[350:, 200:600]
R = roi[:, :, 2].astype(float)
G = roi[:, :, 1].astype(float)
B = roi[:, :, 0].astype(float)

score = np.mean(R / (G + B + 1))

hsv_2 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lower1 = np.array([0, 80, 80])
upper1 = np.array([10, 255, 255])

lower2 = np.array([170, 80, 80])
upper2 = np.array([180, 255, 255])
hsv = hsv_1
mask = cv2.inRange(hsv, lower1, upper1)
mask |= cv2.inRange(hsv, lower2, upper2)
score = np.count_nonzero(mask)

hsv = hsv_2
mask = cv2.inRange(hsv, lower1, upper1)
mask |= cv2.inRange(hsv, lower2, upper2)
score = np.count_nonzero(mask)
plt.imshow(img_1)
plt.show(block=True)

cv2.imshow('hsv_1', hsv_1)
cv2.imshow('hsv_2', hsv_2)

score = np.count_nonzero(mask)
cv2.imshow('img_1', roi)
cv2.imshow('img_2', img_2)

cv2.imshow('diff', cv2.absdiff(hsv_1, hsv_2))
cv2.waitKey(0)