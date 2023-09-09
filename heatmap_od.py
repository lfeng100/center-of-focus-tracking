import cv2 as cv
import numpy as np
from queue import PriorityQueue
import sys
from collections import deque
# from overlay_image import overlay_image_alpha

# An attempt to combine the heatmap and object detecting/trackign approach
# See heatmap.py and object_tracking.py
# Accomplished by applying heatmap over only a desired region decided by the detected bounding boxes
# THIS IS A WORK IN PROGRESS
# Main issue is overlaying the heatmap over the bounding boxes
# NOTE: here we do not provide the option for an accumulated image as in heatmap.py

args = sys.argv[1:]
if args:
    inputfile = args[0]
else:
    inputfile = "basketball_moving.mp4"

cap = cv.VideoCapture("basketball_moving.mp4")
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Tunable Parameters
roi_top_margin = 0 # track inside margin
roi_side_margin = 0 # track inside margin
bounding_box_margin = 2 # scale bounding boxes (do not need accuracy)
min_contour_area = 500 # minimum size of contour 
max_contour_area = 1000 # maximum size of contour 
num_bounding_boxes = 3 # number of bounding boxes in a frame
# BackgroundSubtractor Parameters
od_history = 125 
od_varThreshold = 40

# Tunable Parameters for Heatmap
length = min(10000, frame_count)
accumulate = False
# if accumulate is False, we only stack the previous use_previous terms
use_previous = 5
threshold, maxValue = 20, 20 # threshold sensitivity
output_video = False

# Object detection (better for Stable camera)
object_detector = cv.createBackgroundSubtractorMOG2(
    history=od_history, # number of previous frames to use
    varThreshold=od_varThreshold # detection sensitivity
    )

# Heatmap Subtractor
background_subtractor = cv.bgsegm.createBackgroundSubtractorMOG()
first_iteration_indicator = 1

for i in range(0, length-1):
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # print(height, width)

    # Region of Interest 
    # Should be automated
    y1, y2 = int(height*roi_top_margin), int(height*(1-roi_top_margin))
    x1, x2 = int(width*roi_side_margin), int(width*(1-roi_side_margin))

    roi = frame[y1:y2, x1:x2]

    # Object Detection (FOR STABLE CAMERA)
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

            if (width > height * 3): # prevent long objects (width)
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

    # keep detected bounding boxes in an array to then apply heatmap
    detections_list = []
    for i in range(detections.qsize()):
        size, coords = detections.get()
        detections_list.append(coords)

    # cv.imshow("roi", roi)
    # cv.imshow("Frame", frame)
    # cv.imshow("Mask", mask)

    # If first frame, we start with blank canvas (empty queue)
    if first_iteration_indicator == 1:
        height, width = frame.shape[:2]
        prev_frames = deque()
        prev_frames.append(np.zeros((height, width), np.uint8))
        first_iteration_indicator = 0
    else:
        filter = background_subtractor.apply(frame)  # remove the background
        ret, th1 = cv.threshold(filter, threshold, maxValue, cv.THRESH_BINARY)

        # add to X previous frames where X is defined in use_previous
        if i >= use_previous:
            prev_frames.popleft()
        prev_frames.append(th1)

        new_image = cv.add(prev_frames[0], prev_frames[1])
        
        for j in range(2, len(prev_frames)):
            new_image = cv.add(new_image, prev_frames[j])
        # cv.imshow("new_image", new_image)

        color_image = cv.applyColorMap(new_image, cv.COLORMAP_HOT)
        result_overlay = frame
        result_overlay = cv.cvtColor(result_overlay, cv.COLOR_RGB2RGBA)

        # We output the heatmap only over the bounding box regions
        for box in detections_list:
            # print(box)
            x,y,w,h = box # get coordinates/size
            cv.rectangle(result_overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
            box_roi = color_image[y:y+h, x:x+w]
            # Keep only 254-255 (where 255 is white)
            ret, box_th = cv.threshold(box_roi, 190, 255, cv.THRESH_BINARY)

            # convert box threshold image to include alpha opacity channel
            box_th = cv.cvtColor(box_th, cv.COLOR_RGB2RGBA)

            # overlap over the entire frame
            result_overlay[y:y+box_th.shape[0], 
                           x:x+box_th.shape[1]] = box_th
            
            y1, y2 = y, y + box_th.shape[0]
            x1, x2 = x, x + box_th.shape[1]
            alpha_s = 0.9
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                result_overlay[y1:y2, x1:x2, c] = (alpha_s * box_roi[:, :, c] 
                                                   + alpha_l * result_overlay[y1:y2, x1:x2, c])

            # cv.imshow("result_overlay", box_th)

        cv.imshow("result_overlay", result_overlay)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv.destroyAllWindows()