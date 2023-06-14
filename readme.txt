
					***************************************************************************************
					* This is the readme file for 2023 EE443 Final project Multi-oject tracking tracks.   *
					* Please read the following message carefully!					      *
					* Please contact Hsiang-Wei Huang (hwhuang@uw.edu) if you have any problem or issue.  *
					***************************************************************************************

========================================================================================================================================================================================
## Dataset

The dataset consists of 5 different cameras from c071 ~ c075. The scenario for these cameras is a hospital with 5 different Nvidia Omniverse avatars. Your goal is to track these 
avatars by assigning them with a consistent Tracking ID and also provide the bounding boxes location for these avatars at each frame.

The folder c071 ~ c075 contains the image/video from each cameras as well as label for the train/validation set. The total number of image for each camera is 18010 (from 0~18009).
We also provide a top-down view map (map.png). Feel free to use it if needed.

For camera 71, 72, 73, these are the "training set". You can use the data for any kinds of training (detector, ReID model...). 
The label is provided in each folder with the format of: <frame ID>, <Tracking ID>, <x>, <y>, <w>, <h>, 1, -1, -1, -1 (You can ignore the last four number)

For camera 74, this is the "validation set". You can evaluate your tracking algorithm with this set. Please refer to the evaluation folder for evaluation tutorial.

For camera 75, this is the "test set". You will have no assess to the ground truth file, and we will conduct the evaluation to determine your score and ranking. 
You will need to provide your tracking result on camera 75 in the same formant as "example_tracking_result.txt"

The format is <camera ID>, <tracking ID>, <frame ID>, <x>, <y>, <w>, <h>, -1, -1 
Since the test set is camera 75, the camera ID for your final submission will always be 75!
========================================================================================================================================================================================
## Detection and Embedding

In case you do not want to train your own detector and ReID model, we also provided the following files for you to save some time!

The detection.txt is the detection file after we applied a detector yolov7, and the embedding.npy file is the embedding feature file which contains the detections' embedding features.

Please refer to tutorial.ipynb about how to use these files!
========================================================================================================================================================================================
## Evaluation

Your final score will be evaluated based on 
1. Final Report (15%) + Code (10%)
2. Demo (10%)
3. Ranking (5%) + extra points

The final report and your code should be clearly written with comments.
The demo will be around 15 mins (10 mins of method presentation and 5 mins of video demo results).
The final ranking will be based on the performance of your IDF1 score, which calculates the ratio for your correct tracking ID assignment.
========================================================================================================================================================================================
