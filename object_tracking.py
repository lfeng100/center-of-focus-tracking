import cv2 as cv
from tracker import *
from queue import PriorityQueue
import sys

# Idea: detect bounding boxes over objects of interest
# Run file with: python object_tracking.py
# Change inputfile parameter for different footage
# Exit the program by pressing 'q'

inputfile = "basketball_marshall.mp4"
cap = cv.VideoCapture(inputfile)

# Object tracker
tracker = EuclideanDistTracker()

# Tunable Parameters
roi_top_margin = 1/4 # do not track top/bottom margins (when 0, we track everything)
roi_side_margin = 0 # do not track left/right margins (when 0, we track everything)
bounding_box_margin = 2 # scale bounding boxes (since we do not need accurate bounding boxes)
min_contour_area = 750 # minimum size of contour
max_contour_area = 1500 # maximum size of contour
num_bounding_boxes = 3 # number of bounding boxes in a frame
# BackgroundSubtractor Parameters
od_history = 125 
od_varThreshold = 40

# Track detected regions by maintaining an id
enable_tracking = False

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(
    history=od_history, # number of previous frames to use
    varThreshold=od_varThreshold # detection sensitivity
    )

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape # size of output footage
    # print(height, width)

    # Region of Interest (exclude margins for tracking)
    # Try to automate eventually
    y1, y2 = int(height*roi_top_margin), int(height*(1-roi_top_margin))
    x1, x2 = int(width*roi_side_margin), int(width*(1-roi_side_margin))

    roi = frame[y1:y2, x1:x2]

    # Object Detection (WORKS WELL ONLY WITH STABLE CAMERA)
    mask = object_detector.apply(roi)
    
    # Remove Noise (shadows) 
    # Keep only 254-255 (where 255 is white)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    detections = PriorityQueue(num_bounding_boxes + 1)
    for cnt in contours:
        # Calculate area of contours and remove small elements
        area = cv.contourArea(cnt)
        if area > min_contour_area and area < max_contour_area:
            # cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            # print([x, y, w, h])

            add_bounding_box = True
            if (w > h * 2.5): # prevent long objects (width)
                add_bounding_box = False
                
            if detections.full():
                size, coords = detections.queue[0]
                if area < size:
                    add_bounding_box = False
                else:
                    detections.get() # discard minimum bounding box

            # add bounding box only if larger than current minimum bounding box
            if add_bounding_box:    
                detections.put((
                    area,
                    [max(0, int(x - bounding_box_margin * w)), 
                    max(0, int(y - bounding_box_margin * h)), 
                    int(w + bounding_box_margin * w), 
                    int(h + bounding_box_margin * h)]))

    detections_list = []
    for i in range(detections.qsize()):
        size, coords = detections.get()
        detections_list.append(coords)

    if enable_tracking:
        # Feed detected objects to Object Tracker
        # Get Euclidean Distance

        boxes_ids = tracker.update(detections_list)

        for box_id in boxes_ids:
            x, y, w, h, id = box_id

            # label object
            # cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    else:
        for box in detections_list:
            x,y,w,h = box
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    cv.imshow("Frame", frame)
    # cv.imshow("roi", roi)
    # cv.imshow("Mask", mask)

    # waitKey returns 32 bit integer
    # we only take the least significant 8 bits for the unicode
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()