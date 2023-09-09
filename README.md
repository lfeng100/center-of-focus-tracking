# Description
Code for algorithms used for Research Project with Dr. Lesley Istead. 
Research pertains to developing an generalized algorithm to detect the center-of-action for sports footage.
Please read the `Outline.docx` document for details. 
The code has been briefly annotated. Start with heatmap.py or object_tracking.py. An incomplete combined approach (heatmap + object detection) in heatmap_od.py.

### These algorithms have not been tested thoroughly and should be used as reference/guideline for future more complete implementations.

# Dependencies
Each program was ran in a conda environment. See `environment.yml` for dependencies. If you encounter versioning issues, the main libraries used is cv2 (opencv) and numpy, which may be installed manually.

# Additional Files
The *.mp4 files are example input footage used for the programs, the code should work on any input (with varying degrees of success). The example_output folder contains some heatmap output from these example footage. experimental_tracking folder contains code for object detection/tracking where the objects of interest must be kown beforehand (drawn).

# Potential papers of interest
- https://arxiv.org/abs/2001.05238
- https://www.cs.cmu.edu/~yaser/SheikhJavedKanade_ICCV_2009.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0167865517300260
- https://www.cs.toronto.edu/~davidj/projects/towards_real_time_detection_tracking.pdf
- https://www.researchgate.net/publication/338086954_Use_of_Machine_Learning_to_Automate_the_Identification_of_Basketball_Strategies_Using_Whole_Team_Player_Tracking_Data

