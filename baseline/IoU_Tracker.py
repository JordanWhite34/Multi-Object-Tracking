# This is the baseline code for the single camera tracker using bounding box IoU (intersection over union)
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# calculate the overlap ratio of two bounding boxes
def calculate_iou(bbox1, bbox2):

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    return iou

# base class for tracklet
class tracklet:
    def __init__(self,tracking_ID,box,feature,time):
        self.ID = tracking_ID
        self.boxes = [box]
        self.features = [feature]
        self.times = [time]

        self.cur_box = box
        self.cur_feature = None
        self.alive = True

        self.final_features = None
    
    def update(self,box,feature,time):
        self.cur_box = box
        self.boxes.append(box)
        self.cur_feature = feature
        self.features.append(feature)
        self.times.append(time)
    
    def close(self):
        self.alive = False
    
    def get_avg_features(self):
        self.final_features = sum(self.features)/len(self.features) 
        return self.final_features


# class for multi-object tracker
class tracker:
    def __init__(self):
        self.all_tracklets = []
        self.cur_tracklets = []

    def run(self, detections, features):
        
        for frame_id in range(0,18010):

            if frame_id%1000 == 0:
                print('Tracking | cur_frame {} | total frame 18010'.format(frame_id))

            inds = detections[:,1] == frame_id
            cur_frame_detection = detections[inds]
            cur_frame_features = features[inds]
            
            # no tracklets in the first frame
            if len(self.cur_tracklets) == 0:
                for idx in range(len(cur_frame_detection)):
                    new_tracklet = tracklet(len(self.all_tracklets)+1,cur_frame_detection[idx][3:7],cur_frame_features[idx],frame_id)
                    self.cur_tracklets.append(new_tracklet)
                    self.all_tracklets.append(new_tracklet)
            
            else:
                iou_cost_matrix = np.zeros((len(self.cur_tracklets),len(cur_frame_detection)))

                for i in range(len(self.cur_tracklets)):
                    for j in range(len(cur_frame_detection)):
                        iou_cost_matrix[i][j] = 1 - calculate_iou(self.cur_tracklets[i].cur_box, cur_frame_detection[j][3:7])
                
                if len(cur_frame_features) > 0:
                    det_features = np.asarray([cur_frame_features[j] for j in range(len(cur_frame_detection))], dtype=np.float64)
                    track_features = np.asarray([track.get_avg_features() for track in self.cur_tracklets], dtype=np.float64)

                    # Ensure both feature matrices are 2-dimensional
                    if len(det_features.shape) == 1:
                        det_features = det_features.reshape(1, -1)
                    if len(track_features.shape) == 1:
                        track_features = track_features.reshape(1, -1)

                    feature_cost_matrix = np.maximum(0.0, cdist(track_features, det_features, 'cosine'))  

                    # Fuse cost matrices
                    cost_matrix = 0.325 * iou_cost_matrix + 0.675 * feature_cost_matrix

                    row_inds,col_inds = linear_sum_assignment(cost_matrix)

                    matches = min(len(row_inds),len(col_inds))

                    for idx in range(matches):
                        row,col = row_inds[idx],col_inds[idx]
                        if cost_matrix[row,col] == 1:
                            self.cur_tracklets[row].close()
                            new_tracklet = tracklet(len(self.all_tracklets)+1,cur_frame_detection[col][3:7],cur_frame_features[col],frame_id)
                            self.cur_tracklets.append(new_tracklet)
                            self.all_tracklets.append(new_tracklet)
                        else:
                            self.cur_tracklets[row].update(cur_frame_detection[col][3:7],cur_frame_features[col],frame_id)

                    # initiate unmatched detections as new tracklets
                    for idx,det in enumerate(cur_frame_detection):
                        if idx not in col_inds: # if it is not matched in the above Hungarian algorithm stage
                            new_tracklet = tracklet(len(self.all_tracklets)+1,det[3:7],cur_frame_features[idx],frame_id)
                            self.cur_tracklets.append(new_tracklet)
                            self.all_tracklets.append(new_tracklet)
            
            self.cur_tracklets = [trk for trk in self.cur_tracklets if trk.alive]            

        final_tracklets = self.all_tracklets

        # calculate an average final features (512x1) for all the tracklets
        for trk_id in range(len(final_tracklets)):
            final_tracklets[trk_id].get_avg_features()

        return final_tracklets