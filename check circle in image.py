import cv2
import imutils
import numpy as np
import time


path = 'C:\\Users\\Admin\\Desktop\\github stuff\Millikan Oil Drop\\video and images\\frame1.jpg'
frame = cv2.imread(path)
height, width, _ = frame.shape
# width = frame.get(cv2.CAP_PROP_FRAME_WIDTH)
frame = frame[120:520, 0:int(width)]
height, width, _ = frame.shape


grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


orient = True
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(grayFrame, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
# minimum number of votes (intersections in Hough grid cell)
threshold = 15
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
x1, y1, x2, y2 = lines[1][0]
m = (y2-y1)/(x2-x1)
theta = np.arctan(m)*180/np.pi
grayFrame = imutils.rotate(grayFrame, theta)
frame = imutils.rotate(frame, theta)
blurFrame = cv2.GaussianBlur(grayFrame, (13, 13), 0)


cv2.imshow('original', frame)
cv2.imshow('blur', blurFrame)
circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT,
                           2, 20, param1=30, param2=10, minRadius=0, maxRadius=7)
if circles is not None:
    circles = np.uint16(np.around(circles))
    # circle_number = 0
    j = len(circles[0])
    k = 0
    for i in circles[0, :]:
        if i[1] < height/2:
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 3)
            continue
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255/(j-k)), 3)
        if k == 1:
            break

cv2.imshow("frame", frame)
cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
