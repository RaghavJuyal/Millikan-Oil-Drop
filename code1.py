import cv2
import imutils
import numpy as np
import time
import dlib

# videoCapture = cv2.VideoCapture(0) # For laptop camera
videoCapture = cv2.VideoCapture(
    'C:\\Users\\Admin\\Desktop\\github stuff\Millikan Oil Drop\\video and images\\vid.mp4')  # For video have to use full path
# videoCapture = cv2.VideoCapture(1) # For camera

height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)

number_of_circles = 5

# Stores circle data(location and radius) from the previous frame
prevCircle = [None]*number_of_circles
strike = []

# Function that calculates square of distance b/w circles of 2 frames


def dist(x1, y1, x2, y2): return (int(x1)-int(x2))**2 + (int(y1)-int(y2))**2


# Function that finds the angle to orient the frame
def orient_frame(grayFrame):
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(grayFrame, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    angular_resolution = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, angular_resolution, threshold, np.array([]),
                            min_line_length, max_line_gap)
    x1, y1, x2, y2 = lines[1][0]
    m = (y2-y1)/(x2-x1)
    return np.arctan(m)*180/np.pi


newcircle = True  # Variable for finding new circle
count = 0  # Variable for debugging
orient = False  # Makes sure to find angle to orient frame only once
theta = 0  # Angle to orient frame
count2 = 1  # Variable for debugging'

trackers = []

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    # For Debugging: Time delay to check if code is working properly
    # time.sleep(1)

    frame = frame[120:520, 0:int(width)]  # For cropping the video
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Finding angle to orient the frame
    if not orient:
        orient = True
        theta = orient_frame(grayFrame)

    # Orienting the frame by rotating by theta
    grayFrame = imutils.rotate(grayFrame, theta)
    frame = imutils.rotate(frame, theta)
    blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(frame, (20, 40), (332, 370), (255, 0, 0), 1)
    # cv2.rectangle(frame, (22, 40), (332, 370), (255, 0, 0), 1)
    # Finding a circle in first frame
    if newcircle:
        # print("Frame 1")  # For debugging purposes
        trackers = []
        strike = []
        # Finds all circles
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 2, 20,
                                   param1=30, param2=10, minRadius=0, maxRadius=7)

        # Selects best circle
        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = [None]*number_of_circles
            check = [-1]
            for circle_number in range(0, number_of_circles):
                count_circle = 0
                for i in circles[0, :]:

                    # cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 3)
                    # If at edge skip
                    # (i[0] < i[2] or i[1] < i[2]) or
                    if i[0] < i[2] or i[1] < i[2] or i[0]-i[2] < 20 or i[0]+i[2] > 332 or i[1]-i[2] < 40 or i[1]+i[2] > 370:
                        continue
                    # cv2.rectangle(frame, (20, 40), (332, 370), (255, 0, 0), 4)

                    if chosen[circle_number] is None:
                        if count_circle in check:
                            count_circle += 1
                            continue
                        chosen[circle_number] = i
                        check.append(count_circle)
                        break

                    count_circle += 1
                prevCircle[circle_number] = chosen[circle_number]
                circle_number += 1

            for i in range(0, number_of_circles):
                # print(i)
                # print(chosen)
                if chosen[i] is None:
                    continue
                x1 = chosen[i][0]-chosen[i][2]
                y1 = chosen[i][1]-chosen[i][2]
                x2 = chosen[i][0]+chosen[i][2]
                y2 = chosen[i][1]+chosen[i][2]
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                tkr = dlib.correlation_tracker()
                rect = dlib.rectangle(x1, y1, x2, y2)
                try:
                    tkr.start_track(rgb_frame, rect)
                except:
                    continue
                trackers.append(tkr)
                strike.append(0)
            newcircle = False
        continue

    # Tracking circles

    if not newcircle:
        if count % 5 == 0 and count != 0:
            circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 2, 20,
                                       param1=30, param2=10, minRadius=0, maxRadius=7)
            if circles is not None:
                circles = np.uint16(np.around(circles))
            else:
                newcircle = True
                print("NO CIRCLE")

        bob = 0
        # cv2.rectangle(frame, (20, 40), (332, 370), (255, 0, 0), 4)
        for tracker in trackers:
            a = tracker.update(rgb_frame)
            position = tracker.get_position()
            x1 = int(position.left())
            y1 = int(position.top())
            x2 = int(position.right())
            y2 = int(position.bottom())
            still_there = 1
            if count % 5 == 0 and count != 0:
                still_there = 0
                if circles is not None:
                    for i in circles[0, :]:
                        if i[0] <= x1-1 or i[0] >= x2+1 or i[1] <= y1-1 or i[1] >= y2+1:
                            continue
                        else:
                            still_there = 1
                            break
                else:
                    still_there = 0
            if still_there == 0:
                strike[bob] += 1
            if x1 < 19 or y1 < 39 or x2 > 333 or y2 > 371 or strike[bob] == 3:
                newcircle = True
                if still_there == 0:

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # trackers = []
                # break
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            bob += 1

    # print('\n')
    cv2.imshow("frame", frame)  # Display frame
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
