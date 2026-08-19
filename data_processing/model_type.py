import numpy as np

class Model:
    def __init__(self):
        self.name = None, 
        self.keypoints = []
        self.minimal_set = [    
        # Body (17 COCO-style keypoints)
        "nose", "left_eye", "right_eye",
          "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        # "left_hip", "right_hip",
        # "left_knee", "right_knee", "left_ankle", "right_ankle",

        # Left hand (21 keypoints, MediaPipe-style)
        # "left_hand_wrist",
        "left_hand_thumb1",
        #   "left_hand_thumb2", "left_hand_thumb3", "left_hand_thumb4",
        "left_hand_index1",
        #   "left_hand_index2", "left_hand_index3", "left_hand_index4",
        "left_hand_middle1",
        #   "left_hand_middle2", "left_hand_middle3", "left_hand_middle4",
        "left_hand_ring1",
        #   "left_hand_ring2", "left_hand_ring3", "left_hand_ring4",
        "left_hand_pinky1",
        #   "left_hand_pinky2", "left_hand_pinky3", "left_hand_pinky4",

        # Right hand (21 keypoints, MediaPipe-style)
        # "right_hand_wrist",
        "right_hand_thumb1",
        #   "right_hand_thumb2", "right_hand_thumb3", "right_hand_thumb4",
        "right_hand_index1",
        #   "right_hand_index2", "right_hand_index3", "right_hand_index4",
        "right_hand_middle1",
        #   "right_hand_middle2", "right_hand_middle3", "right_hand_middle4",
        "right_hand_ring1",
        #   "right_hand_ring2", "right_hand_ring3", "right_hand_ring4",
        "right_hand_pinky1",
        #   "right_hand_pinky2", "right_hand_pinky3", "right_hand_pinky4"
        ]
        

class WholeBody(Model):
    def __init__(self):
        super().__init__()
        self.name = "WholeBody"
        self.keypoints = [
            # Body (17 COCO-style keypoints)
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",

            # Feet (6 keypoints)
            "left_big_toe", "left_small_toe", "left_heel",
            "right_big_toe", "right_small_toe", "right_heel",

            # Face (68 keypoints, standard landmark order)
            "face_0", "face_1", "face_2", "face_3", "face_4", "face_5", "face_6",
            "face_7", "face_8", "face_9", "face_10", "face_11", "face_12", "face_13",
            "face_14", "face_15", "face_16", "face_17", "face_18", "face_19", "face_20",
            "face_21", "face_22", "face_23", "face_24", "face_25", "face_26", "face_27",
            "face_28", "face_29", "face_30", "face_31", "face_32", "face_33", "face_34",
            "face_35", "face_36", "face_37", "face_38", "face_39", "face_40", "face_41",
            "face_42", "face_43", "face_44", "face_45", "face_46", "face_47", "face_48",
            "face_49", "face_50", "face_51", "face_52", "face_53", "face_54", "face_55",
            "face_56", "face_57", "face_58", "face_59", "face_60", "face_61", "face_62",
            "face_63", "face_64", "face_65", "face_66", "face_67",

            # Left hand (21 keypoints, MediaPipe-style)
            "left_hand_wrist",
            "left_hand_thumb1", "left_hand_thumb2", "left_hand_thumb3", "left_hand_thumb4",
            "left_hand_index1", "left_hand_index2", "left_hand_index3", "left_hand_index4",
            "left_hand_middle1", "left_hand_middle2", "left_hand_middle3", "left_hand_middle4",
            "left_hand_ring1", "left_hand_ring2", "left_hand_ring3", "left_hand_ring4",
            "left_hand_pinky1", "left_hand_pinky2", "left_hand_pinky3", "left_hand_pinky4",

            # Right hand (21 keypoints, MediaPipe-style)
            "right_hand_wrist",
            "right_hand_thumb1", "right_hand_thumb2", "right_hand_thumb3", "right_hand_thumb4",
            "right_hand_index1", "right_hand_index2", "right_hand_index3", "right_hand_index4",
            "right_hand_middle1", "right_hand_middle2", "right_hand_middle3", "right_hand_middle4",
            "right_hand_ring1", "right_hand_ring2", "right_hand_ring3", "right_hand_ring4",
            "right_hand_pinky1", "right_hand_pinky2", "right_hand_pinky3", "right_hand_pinky4"
            ]
        self.minimal_idxs = [idx for idx, key in enumerate(self.keypoints) if key in self.minimal_set]
    
    def get_minimal_keypoints(self, keypoints, thr=0.6):
        idx_to_remove = np.argwhere(keypoints[:, :, 2] < thr)
        keypoints[idx_to_remove[:, 0], idx_to_remove[:, 1], :] = np.nan
        return keypoints[:, self.minimal_idxs, :2]
    
    def get_index(self, name, minimal_set=True):
        marker_set = self.minimal_set if minimal_set else self.keypoints
        if name not in marker_set:
            return None
        return marker_set.index(name)

        