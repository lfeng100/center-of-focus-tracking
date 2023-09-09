import cv2 as cv
import numpy as np
import imutils
import os
import sys

# Run file with: python single_object.py SOME_FOOTAGE.mp4 (must be in same folder)
# if no argument is specified, it will use the default footage specified
# experiment for single object tracking where region of interest must be decided beforehand
# specify inputfile to use different input footage

args = sys.argv[1:]
if args:
    inputfile = args[0]
else: 
    inputfile = "basketball_marshall.mp4"

capture = cv.VideoCapture(inputfile)
# capture = cv.VideoCapture(0) # uncomment to set input to computer camera/webcam device

# background_subtractor = cv.bgsegm.createBackgroundSubtractorMOG()
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

# keep trackers in dictionary (we use only one)
TrDict = {
    'csrt': cv.TrackerCSRT_create,
    'kcf' : cv.TrackerKCF_create,
    'boosting' : cv.legacy.TrackerBoosting_create,
    'mil': cv.TrackerMIL_create,
    'tld': cv.legacy.TrackerTLD_create,
    'medianflow': cv.legacy.TrackerMedianFlow_create,
    'mosse':cv.legacy.TrackerMOSSE_create
    }

# initialize tracker
tracker = TrDict['csrt']()
#tracker = cv.TrackerCSRT_create()

ret, frame = capture.read()
frame = imutils.resize(frame,width=600)
cv.imshow('Frame', frame)

# draw bounding box with region of interest Frame
bb = cv.selectROI('Frame', frame)

# initialize tracker with the frame and bounding box
tracker.init(frame,bb)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=600)
    (success, box) = tracker.update(frame)

    if success:
        # x,y top left corner, w,h: width and height
        (x,y,w,h) = [int(a) for a in box]

        # draw rectangle on frame
        cv.rectangle(frame, (x,y), (x+w,y+h), (100,255,0), 2)
    cv.imshow('Frame', frame)

    # waitKey returns 32 bit integer
    # we only take the least significant 8 bits for the unicode
    key = cv.waitKey(5) & 0xFF 
    if key == ord('q'): # ord gets the unicode representation of 'q'
        break

# release video and windows
capture.release()
cv.destroyAllWindows()