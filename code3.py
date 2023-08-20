import cv2
import imutils
import numpy as np
import time
import dlib


class OilDrop:
    def __init__(self, x, y, radius):
        self.radius = radius  # Should they be integers?
        self.x_current = x  # Always integer
        self.y_current = y  # Always integer
        self.x_previous = x  # Always integer
        self.y_previous = y  # Always integer
        self.x_top_left = round(x - radius)  # Always integer
        self.y_top_left = round(y - radius)  # Always integer
        self.x_bottom_right = round(x + radius)  # Always integer
        self.y_bottom_right = round(y + radius)  # Always integer
        self.tracker = None
        self.rect = None
        self.current_velocity = 0.0  # Current velocity
        self.average_velocity = 0.0  # Average velocity in current direction
        self.average_velocity_up = 0.0  # Average velocity up
        self.average_velocity_down = 0.0  # Average velocity down
        self.previous_direction = 0  # 1 is up, -1 is down, 0 is not defined yet
        self.current_direction = 0  # 1 is up, -1 is down, 0 is not defined yet
        self.frames = 0  # Counts number of frames it has gone in the same direction
        self.frames_up = 0  # Number of frames oil drop went up
        self.frames_down = 0  # Number of frames oil drop went down
        self.has_it_gone_up_and_down = False  # Decides whether we store the information checked in delete tracker
        self.should_delete = False
        self.confidence = None
        self.current_velocity_y = 0  # Always integer
        self.strikes = 0  # How many chances you get for having 0 current y velocity

    def update_average_velocity(self):  # Call after updating frames, updates average velocity, call if direction did not change, what units?
        d = dist(self.x_current, self.y_current, self.x_previous, self.y_previous)
        self.current_velocity = d
        self.average_velocity = ((self.frames * self.average_velocity) + d) / (self.frames + 1)
        self.current_velocity_y = self.y_current - self.y_previous

    def update_direction(self):  # Updates current and previous direction
        self.previous_direction = self.current_direction

        if self.y_current - self.y_previous <= 0:
            self.current_direction = 1
        else:
            self.current_direction = -1

    def did_direction_change(self):  # Checks if there is a change in direction
        if self.current_direction * self.previous_direction < 0:
            return True
        else:
            return False

    def update_frame(self):  # Updates number of frames in same direction
        self.frames += 1

    def update_tracker(self, rgb_frame):
        # previous x,y = current x,y
        self.x_previous = self.x_current
        self.y_previous = self.y_current

        # Update tracker
        self.confidence = self.tracker.update(rgb_frame)
        position = self.tracker.get_position()
        self.x_top_left = int(position.left())
        self.y_top_left = int(position.top())
        self.x_bottom_right = int(position.right())
        self.y_bottom_right = y2 = int(position.bottom())
        self.x_current = round((self.x_top_left + self.x_bottom_right) / 2)
        self.y_current = round((self.y_top_left + self.y_bottom_right) / 2)

        # If at boundary or not perfectly clear, delete tracker
        if self.confidence < confidence_threshold:
            self.should_delete = True

        elif crossed_border(self.x_top_left, self.y_top_left, self.x_bottom_right, self.y_bottom_right):
            self.should_delete = True

        elif self.current_velocity_y == 0:
            self.strikes += 1
            if self.strikes == 3:
                self.should_delete = True

        if not self.should_delete:
            self.update_direction()

            if self.did_direction_change():
                self.has_it_gone_up_and_down = True

                if self.current_direction == 1:  # Went from DOWN to UP
                    self.average_velocity_down = ((self.average_velocity_down * self.frames_down)+ (self.average_velocity * self.frames)) / (self.frames_down + self.frames)
                    self.frames_down += self.frames
                elif self.current_direction == -1:  # Went from UP to DOWN
                    self.average_velocity_up = ((self.average_velocity_up * self.frames_up)+ (self.average_velocity * self.frames)) / (self.frames_up + self.frames)
                    self.frames_up += self.frames
                else:
                    # It shouldnt come here
                    print("HOLD UP It shouldnt be here")

                self.current_velocity = dist(self.x_current, self.y_current, self.x_previous, self.y_previous)
                self.average_velocity = self.current_velocity
                self.frames = 1
            else:
                self.update_average_velocity()
                self.update_frame()

        # Update direction
        # Update check did direction change
        # if no:
        # update frame
        # update average velocity
        # else
        # self.has_it_gone_up_and_down = True maybe update in direction change
        # depending on direction:
        # update up/down average velocity
        # update up/down frame
        # new average velocity is dist()
        # frames are now 1
        pass

    def create_tracker(self, rgb_frame):
        self.tracker = dlib.correlation_tracker()
        self.rect = dlib.rectangle(self.x_top_left, self.y_top_left, self.x_bottom_right, self.y_bottom_right)
        try:
            self.tracker.start_track(rgb_frame, self.rect)
        except:
            print("Could not create tracker")

    def delete_tracker(self):  # check if up and down, if yes store data else dont store data, then delete tracker
        if self.current_direction == -1:  # Including current down velocity
            self.average_velocity_down = ((self.average_velocity_down * self.frames_down)+ (self.average_velocity * self.frames)) / (self.frames_down + self.frames)
            self.frames_down += self.frames
        elif self.current_direction == 1:  # Including current up velocity
            self.average_velocity_up = ((self.average_velocity_up * self.frames_up)+ (self.average_velocity * self.frames)) / (self.frames_up + self.frames)
            self.frames_up += self.frames
        else:
            # It shouldnt come here
            print("HOLD UP It shouldnt be here")

        if self.has_it_gone_up_and_down:
            if (self.average_velocity_up > min_velocity and self.average_velocity_down > min_velocity) and (self.frames_up + self.frames_down >= min_frames_for_valid_drop):
                final_oildrop_values.append((self.average_velocity_up, self.average_velocity_down, self.radius))
        else:
            pass


# videoCapture = cv2.VideoCapture(0) # For laptop camera
videoCapture = cv2.VideoCapture("C:\\Users\\Admin\\Desktop\\github stuff\Millikan Oil Drop\\video and images\\vid.mp4")  # For video have to use full path
# videoCapture = cv2.VideoCapture(1) # For camera
height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
max_number_of_circles = 5
prevCircle = [None] * max_number_of_circles
new_circle_frames = 5  # After these many frames we search for new circles
left_border = 20
right_border = 332
top_border = 40
bottom_border = 370
min_gap_for_new_oil_drop = 10  # pixels
confidence_threshold = 0  # Change it
min_velocity = 1e-6
min_frames_for_valid_drop = 12
max_velocity = 100
final_oildrop_values = []
min_line_coord = None
max_line_coord = None
distance_between_lines = None


def main():
    total_frames = 0  # Frames passed till now
    have_aligning_angle = False  # Makes sure to find angle to orient frame only once
    aligning_angle = 0  # Angle to align frame
    oildrops = []  # Stores the OilDrop object for each oil drop
    oildrops_current_location = []  # Stores current x,y of each oil drops being tracked
    found_distance = False
    # videocapture
    # while in loop:
    start = time.time()
    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame = frame[120:520, 0 : int(width)]  # For cropping the frame
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not have_aligning_angle:  # if not oriented orient
            have_aligning_angle = True
            aligning_angle = align_frame_angle(grayFrame)

        # Orienting the frame by rotating by theta
        grayFrame = imutils.rotate(grayFrame, aligning_angle)
        frame = imutils.rotate(frame, aligning_angle)

        blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)  # Used for getting circles
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Used for trackers
        cv2.rectangle(frame,(left_border, top_border),(right_border, bottom_border),(255, 0, 0), 1,)
        if not found_distance:
            
            min_line_coord,max_line_coord, distance_between_lines= calculate_distance_between_lines(grayFrame=grayFrame)
            found_distance = True
        # cv2.rectangle(frame,(min_line_coord[0],min_line_coord[1]),(min_line_coord[2],min_line_coord[3]),(0, 0, 255),2,) # For checking if lines are correct
        # cv2.rectangle(frame,(max_line_coord[0],max_line_coord[1]),(max_line_coord[2],max_line_coord[3]),(0, 0, 255),2,)
        # update tracker stuff
        oildrops_current_location = []
        for idx in reversed(range(len(oildrops))):
            oildrop = oildrops[idx]
            oildrop.update_tracker(rgb_frame=rgb_frame)
            if oildrop.should_delete:
                oildrop.delete_tracker()
                del oildrops[idx]
            else:
                oildrops_current_location.append((oildrop.x_current, oildrop.y_current))
                cv2.rectangle(frame,(oildrop.x_top_left, oildrop.y_top_left),(oildrop.x_bottom_right, oildrop.y_bottom_right),(255, 0, 0),2,)

        if (total_frames % new_circle_frames == 0):  # Check for new circles after every 10 frames
            # Maybe make new circle function
            circles = cv2.HoughCircles(blurFrame,cv2.HOUGH_GRADIENT,2,20,param1=30,param2=10,minRadius=0,maxRadius=7,)

            if circles is None:
                print("Pump more")
            else:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    is_new_drop = True
                    # print(circle[0], circle[1], circle[2])
                    # print("\n")
                    for oildrop in oildrops_current_location:  # If already tracked
                        # print(oildrop[0], oildrop[1])
                        if (dist(circle[0], circle[1], oildrop[0], oildrop[1]) <= min_gap_for_new_oil_drop):
                            is_new_drop = False
                            break

                    if is_new_drop and (circle[0] < circle[2] or circle[1] < circle[2]):  # Check if at edge of screen by comparing location and radius
                        is_new_drop = False

                    if is_new_drop and crossed_border(circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2], circle[1] + circle[2],):
                        is_new_drop = False

                    if not is_new_drop:
                        continue
                    else:
                        new_drop = OilDrop(circle[0], circle[1], circle[2])
                        oildrops.append(new_drop)
                        oildrops_current_location.append((new_drop.x_current, new_drop.y_current))
                        cv2.rectangle(frame,(new_drop.x_top_left, new_drop.y_top_left),(new_drop.x_bottom_right, new_drop.y_bottom_right),(0, 255, 0),2,)  # Change colour for new circle
                        new_drop.create_tracker(rgb_frame=rgb_frame)

        # every 10 frames find all circles
        # if they already have tracker(check if distance is close enough), ignore them else create new [have list of objects]
        # maybe consider that we get 4 locations for bounding box. maybe bounding box in class function
        # Have function for border for search and for track[only search for circles in this region][if outside dont search]

        # total frames+=1
        # if end or press q then end
        # maybe if they press g then display graph else display at end
        total_frames += 1
        cv2.imshow("frame", frame)  # Display frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    end = time.time()
    print(f"Time: {end-start}")
    print(f"Frames: {total_frames}")
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Distance between max lines = {distance_between_lines}")
    print(final_oildrop_values)
    print("HI")
    print(len(final_oildrop_values))


# Function that calculates distance b/w (x1,y1) and (x2,y2)
def dist(x1, y1, x2, y2):
    return ((int(x1) - int(x2)) ** 2 + (int(y1) - int(y2)) ** 2) ** 0.5


def crossed_border(x1, y1, x2, y2):
    if x1 < left_border or y1 < top_border or x2 > right_border or y2 > bottom_border:
        return True
    else:
        return False


# Function that finds the angle to align the frame
def align_frame_angle(grayFrame):
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
    lines = cv2.HoughLinesP(edges, rho, angular_resolution, threshold, np.array([]), min_line_length, max_line_gap,)
    x1, y1, x2, y2 = lines[1][0]
    m = (y2 - y1) / (x2 - x1)
    return np.arctan(m) * 180 / np.pi


def calculate_distance_between_lines(grayFrame):
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
    lines = cv2.HoughLinesP(edges, rho, angular_resolution, threshold, np.array([]), min_line_length, max_line_gap,)
    x1min, y1min, x2min, y2min = lines[0][0]
    x1max, y1max, x2max, y2max = lines[0][0]
    for line in lines[1, :]:
        x1, y1, x2, y2 = line
        if y1 > y1max:
            x1max, y1max, x2max, y2max = (x1, y1, x2, y2)
        if y1 < y1min:
            x1min, y1min, x2min, y2min = (x1, y1, x2, y2)

    return [[x1min, y1min, x2min, y2min], [x1max, y1max, x2max, y2max], y1max - y1min]

    print(lines)


if __name__ == "__main__":
    main()
