import cv2 as cv
import numpy as np
# import imutils
import os
import sys

# Run file with: python multiple_object.py SOME_FOOTAGE.mp4 (must be in same folder)
# if no argument is specified, it will use the default footage specified
# experiment for multiple object tracking where region of interest must be decided beforehand
# specify inputfile to use different input footage

args = sys.argv[1:]
if args:
    inputfile = args[0]
else:
    inputfile = "basketball_marshall.mp4"

capture = cv.VideoCapture(inputfile)
# background_subtractor = cv.bgsegm.createBackgroundSubtractorMOG()
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

# keep trackers in dictionary (only use one)
TrDict = {
    'csrt': cv.legacy.TrackerCSRT_create,
    'kcf' : cv.TrackerKCF_create,
    'boosting' : cv.legacy.TrackerBoosting_create,
    'mil': cv.TrackerMIL_create,
    'tld': cv.legacy.TrackerTLD_create,
    'medianflow': cv.legacy.TrackerMedianFlow_create,
    'mosse':cv.legacy.TrackerMOSSE_create
    }

# initialize tracker
trackers = cv.legacy.MultiTracker_create()

ret, frame = capture.read()

# add multiple trackers
num_trackers = 2
for i in range(num_trackers):
    cv.imshow('Frame',frame)
    bbi = cv.selectROI('Frame',frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i, frame, bbi)

frameNumber = 2
baseDir = r'./tmp'

while True:
    ret, frame = capture.read()
    if not ret:
        break
    (success, boxes) = trackers.update(frame)
    np.savetxt(baseDir + '/frame_' + str(frameNumber) + '.txt', boxes, fmt='%f')
    frameNumber += 1

    for box in boxes:
        # x,y top left corner, w,h: width and height
        (x, y, w, h) = [int(a) for a in box]

        # draw rectangle on frame
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
    cv.imshow('Frame',frame)

    # waitKey returns 32 bit integer
    # we only take the least significant 8 bits for the unicode
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()