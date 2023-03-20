import cv2 as cv
c = 0
for i in range(100):
    c += 1
    img = cv.imread('white.jpg')
    cv.imwrite(f"data_om/None/{c}.jpg", img)
