import os
import yaml
import numpy as np


class ProcessingOptions:
    def __init__(self):
        self.paired_event_idx = None
        self.set_defaults()
        if not os.path.exists("data_options.yaml"):
        #     self.from_file("data_options.yaml")
        # else:
            self.to_file("data_options.yaml")

    def set_defaults(self):
        self.clusters = {
            "thorax": [
                ["thx_r", "thx_l", "thx_d"],
                ["ster", "xiph", "c7", "t10", "ribs_r", "ribs_l", "clav_sc_r", "clav_sc_l"],
            ],
            "arm_r": [["hum_a_r", "hum_p_r", "hum_d_r"], ["epic_m_r", "epic_lat_r"]],
            "arm_l": [["hum_a_l", "hum_p_l", "hum_d_l"], ["epic_m_l", "epic_lat_l"]],
        }
        self.low_pass_filter = {"enable": True, "cutoff": 6, "order": 4}
        self.fill_gaps = {"enable": True, "threshold": 10}
        self.motion_segmentation = {"threshold_onset": 0.1, "threshold_drinking": 0.1, "threshold_transporting": 0.1}
        self.events = {"cut": True, "names": ["Start", "Stop"]}
        self.CNN_model = {
            "mode": "balanced",
            "device": "cuda",
            "backend": "onnxruntime",
            "save_annotated_images": False,
        }
        self.projection = {"clusters": self.clusters, "replace_existing": False}
        self.jump_filtering = {
            "enable": True,
            "smooth_process_noise": {"opposite_arm": 1e-1, "motion_arm": 8, "center": 10},
            "smooth_measurement_noise": {"opposite_arm": 1e-4, "motion_arm": 1e-3, "center": 1e-2},
            "smooth_nis_threshold": {"opposite_arm": 10, "motion_arm": 40, "center": 5},
            "save_plot": True,
        }
        self.export_trc = True

    def to_file(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self.get_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def get_dict(self):
        return {
            "export_trc": self.export_trc,
            "data_processing": {"low_pass_filter": self.low_pass_filter, "jump_filtering": self.jump_filtering},
            "motion_segmentation": self.motion_segmentation,
            "CNN_model": self.CNN_model,
            "projection": self.projection,
            "fill_gaps": self.fill_gaps,
            "events": self.events,
        }

    def from_file(self, file_path):
        with open(file_path, "r") as f:
            options_dict = yaml.safe_load(f)
        self.from_dict(options_dict)

    def from_dict(self, options_dict):
        for key, value in options_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RGBDProcessingOptions(ProcessingOptions):
    def __init__(self):
        super().__init__()
        self.type = "RGBD"


class ViconProcessorOptions(ProcessingOptions):
    def __init__(self):
        super().__init__()
        self.type = "Vicon"
        self.paired_event_idx = None

    def set_events(self, events, data_rate=100):
        if "LABELS" not in events:
            self.paired_event_idx = None
            return
        events_names = np.array(events["LABELS"]["value"]).astype("U10")
        idxs_starts = np.argwhere(events_names == "Start")[:, 0]
        idxs_stops = np.argwhere(events_names == "Stop")[:, 0]
        values = events["TIMES"]["value"][:, np.union1d(idxs_starts, idxs_stops)]
        factor = values[0] * 60
        event_times = values[1]
        event_times += factor
        paired_event = np.array([event_times[i : i + 2] for i in range(0, (len(event_times) // 2) * 2, 2)])
        paired_event_idx = paired_event * data_rate
        self.paired_event_idx = paired_event_idx.astype(int)


if __name__ == "__main__":
    options = ProcessingOptions()
    options.to_file("data_options.yaml")
