======================================================================================================================================================================
To run the code, please install the required packages including numpy, sklearn, and scipy.

And then, directly run "python main.py". You should be able to obtain a baseline tracking result.
======================================================================================================================================================================
Baseline Method Discription:

The baseline method consists of two parts. 

1. A single camera tracking method using IoU for association. (IoU_Tracker.py)
2. A postprocessing clustering method that merge the tracklets together based on appearance. (Processing.py)
======================================================================================================================================================================
