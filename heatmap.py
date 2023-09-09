import numpy as np
import cv2 as cv
import os
import sys
from make_video import make_video
from progress.bar import Bar
from collections import deque

# Idea: Difference consecutive frames to track movement and objects of interest
# Run file with: python heatmap.py SOME_FOOTAGE.mp4
# SOME_FOOTAGE.mp4 must be in the same folder
# if no argument is specified, it will use the default basketball_marshall.mp4
# NOTE: marshall is a form of stationary camera
# Exit the program by pressing 'q'

def main():
    args = sys.argv[1:]
    if args:
        inputfile = args[0]
        filename = args[0].split(".")[0]
    else: 
        inputfile = "basketball_marshall.mp4"
        filename = "basketball_marshall"

    capture = cv.VideoCapture(inputfile)

    # Set up background subtractor
    # other options:
    # BackgroundSubtractorCNT
    # BackgroundSubtractorLSBP
    background_subtractor = cv.bgsegm.createBackgroundSubtractorMOG()
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Tunable parameters
    # set max number of frames (length of video clip)
    max_frames = 2000
    length = min(max_frames, frame_count)

    # when accumulate is True, we retain the heatmap information from previous frames (like a snail trail)
    accumulate = False

    # if accumulate is False, we only display the previous use_previous terms
    use_previous = 3
    # threshold sensitivity
    use_prev_threshold, use_prev_maxValue = 40, 40 
    output_video = False
    
    # set values for threshold binarization
    if(accumulate):
        threshold = 2
        maxValue = 2
    else:
        threshold = use_prev_threshold
        maxValue = use_prev_maxValue

    try:
        # creating a frames folder
    	if not os.path.exists('frames'):
    		os.makedirs('frames')

    # if not created then raise error
    except OSError:
    	print ('Error: Creating directory of data')

    if output_video:
        bar = Bar('Processing Frames', max=length)

    # for the first frame we do not have previous frames
    first_iteration_indicator = 1

    for i in range(0, length-1):
        ret, frame = capture.read()

        # If first frame, we set up an empty image
        # When accumulate is False, we must set up a queue to remove old frames
        if first_iteration_indicator == 1:
            # first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            if(accumulate):
                accum_image = np.zeros((height, width), np.uint8)
            else:
                prev_frames = deque()
                prev_frames.append(np.zeros((height, width), np.uint8))
            first_iteration_indicator = 0
        else:
            filter = background_subtractor.apply(frame)  # remove the background

            ret, th1 = cv.threshold(filter, threshold, maxValue, cv.THRESH_BINARY)

            if(accumulate):
                # add to the accumulated image
                accum_image = cv.add(accum_image, th1)

                color_image = cv.applyColorMap(accum_image, cv.COLORMAP_HOT)
                result_overlay = cv.addWeighted(frame, 0.7, color_image, 0.7, 0)

                # produce output video
                if output_video:
                    name = "./frames/frame%d.jpg" % i
                    cv.imwrite(name, result_overlay)
                else:
                    # cv.imshow(name, video_frame)
                    cv.imshow("result_overlay", result_overlay)
            else:
                # add to X previous frames where X is defined in use_previous
                if i >= use_previous:
                    prev_frames.popleft()
                prev_frames.append(th1)
                new_image = cv.add(prev_frames[0], prev_frames[1])
                for j in range(2, len(prev_frames)):
                    new_image = cv.add(new_image, prev_frames[j])

                # apply heatmap
                color_image = cv.applyColorMap(new_image, cv.COLORMAP_HOT)
                result_overlay = cv.addWeighted(frame, 0.7, color_image, 0.7, 0)

                # produce output video
                if output_video:
                    name = "./frames/frame%d.jpg" % i
                    cv.imwrite(name, result_overlay)
                else:
                    # cv.imshow(name, frame)
                    cv.imshow("result_overlay", result_overlay)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        if output_video:
            bar.next()
    if output_video:
        bar.finish()
        make_video('./frames/', './output_' + filename + '.avi')

    # save the final heatmap
    # cv.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    capture.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    main()
