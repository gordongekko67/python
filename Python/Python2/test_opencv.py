import cv2
img = cv2.imread('005.jpg')
cv2.imshow('Image', img)
cv2.waitKey(800000)
print (img.shape)
