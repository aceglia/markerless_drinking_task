import os
import numpy as np
import yaml
import ezc3d
import matplotlib.pyplot as plt


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_local_cs(data, name_cluster, side=""):
    list_name = list(data.channel)
    idxs = [list_name.index(side + n) for n in name_cluster]
    marker = data.values[:, idxs, :]
    M2, M1, M3 = marker[:3, 0, :], marker[:3, 1, :], marker[:3, 2, :]
    first_axis_vector = M3 - M1
    second_axis_vector = M2 - M1
    third_axis_vector = -np.cross(first_axis_vector, second_axis_vector, axis=0)
    second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
    n_frames = first_axis_vector.shape[1]
    rt = np.zeros((4, 4, n_frames))
    rt[:3, 0, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
    rt[:3, 1, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
    rt[:3, 2, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
    rt[:3, 3, :] = M1
    rt[3, 3, :] = 1
    return rt


def plot_cs(rt, M1, M2, M3):
    fig = plt.figure("bis")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    rot_m_a = rt
    x_y_z = rot_m_a[:3, 3]
    vecx = rot_m_a[:3, 0]
    vecy = rot_m_a[:3, 1]
    vecz = rot_m_a[:3, 2]
    for marker in [M1, M2, M3]:
        ax.scatter(marker[0, :], marker[1, :], marker[2, :], c="k", marker="o")
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecx[0], vecx[1], vecx[2], length=60, normalize=False, color="r")
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecy[0], vecy[1], vecy[2], length=60, normalize=False, color="g")
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecz[0], vecz[1], vecz[2], length=60, normalize=False, color="b")
    # ax.scatter(cluster_in_Ra[0, 0, ], cluster_in_Ra[ 0, 1,], cluster_in_Ra[ 0, 2,], c='r', marker='o')
    # ax.scatter(cluster_in_Ra[1, 0, ], cluster_in_Ra[ 1, 1,], cluster_in_Ra[ 1, 2,], c='g', marker='o')
    # ax.scatter(cluster_in_Ra[2, 0, ], cluster_in_Ra[ 2, 1,], cluster_in_Ra[ 2, 2,], c='b', marker='o')
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


def get_from_dict(cal_dict, key):
    if key not in cal_dict.keys():
        raise ValueError(f"{key} not found in calibration dict")
    return np.array(cal_dict[key])


def get_marker_from_cluster(data, name_cluster, calibration_dict, marker_local_name, markers=None, markers_names=None):
    rt = get_local_cs(data, name_cluster)
    marker_in_local = get_from_dict(calibration_dict, marker_local_name)
    marker_trans_global = np.zeros((4, 1, data.values.shape[-1]))
    for i in range(data.shape[-1]):
        marker_trans_global[..., 0, i] = np.dot(rt[..., i], marker_in_local)
    if markers is None:
        list_name = list(data.channel.values)
        idxs = [list_name.index(n) for n in markers_names if n in list_name]
        markers = data.values[:, idxs, :]
    markers = np.concatenate((markers, marker_trans_global), axis=1)
    if marker_local_name in markers_names:
        if marker_local_name not in list(data.channel.values):
            marker_local_name_to_save = marker_local_name
        else:
            idx_to_replace = list(data.channel).index(marker_local_name)
            non_visible_frame = np.where(np.isnan(data.values[0, idx_to_replace, :]))[0]
            markers[:, idx_to_replace, non_visible_frame] = marker_trans_global[..., 0, non_visible_frame]
            marker_local_name_to_save = marker_local_name + "_virtual"
    else:
        marker_local_name_to_save = marker_local_name
    markers_names.append(marker_local_name_to_save)
    return markers, markers_names

class ProcessorOptions:
    def __init__(self):
        self.clusters = {"thorax": [["thx_r", "thx_l", "thx_d"], ["ster", "xiph", "c7", "t10", "ribs_r", "ribs_l", "clav_sc_r", "clav_sc_l"]],
                        "arm_r": [["hum_a_r", "hum_p_r", "hum_d_r"], ["epic_m_r", "epic_l_r"]],
                        "arm_l": [["hum_a_l", "hum_p_l", "hum_d_l"], ["epic_m_l", "epic_l_l"]]
                        }
        self.low_pass_filter = True
        self.filter_cutoff = 6
        self.filter_order = 4
        self.fill_gaps = True
        self.fill_gaps_threshold = 10
        self.cut_from_events = True
        self.events = ["Start", "Stop"]
        self.export_trc = True
        if os.path.exists("data_options.yaml"):
            self.from_file("data_options.yaml")

    def from_file(self, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return {
            "clusters": self.clusters,
            "low_pass_filter": self.low_pass_filter,
            "filter_cutoff": self.filter_cutoff,
            "filter_order": self.filter_order,
            "fill_gaps": self.fill_gaps,
            "fill_gaps_threshold": self.fill_gaps_threshold,
            "cut_from_events": self.cut_from_events,
            "events": self.events,
            "export_trc": self.export_trc
        }

    def to_file(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

class ViconProcessor:
    def __init__(self, calibration_files=None):
        self.vicon_data = None
        self.calibration_files = calibration_files
        self.options = ProcessorOptions()
        if self.calibration_files is not None:
            self.initialize()
        self.calibration_data = {}

    def initialize(self, calibration_files=None, options_file=None):
        if self.calibration_files is None and calibration_files is not None:
            self.calibration_files = calibration_files
        if options_file is not None:
            self.options.from_file(options_file)
        if self.calibration_files is not None and len(self.calibration_files) > 0:
            self._load_calibration_data()

    def _load_calibration_data(self):
        for file_path in self.calibration_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Calibration file not found: {file_path}")
            data = ezc3d.c3d(file_path)
            markers_names_init = list(data["parameters"]["POINT"]["LABELS"]["value"])
            marker_names = [name for name in markers_names_init if "*" not in name]
            idx_markers = [markers_names_init.index(name) for name in marker_names]
            markers_data = data["data"]["points"][:3, idx_markers, :]
            unit = data["parameters"]["POINT"]["UNITS"]["value"][0]
            if unit != "m":
                markers_data /= 1000.0
            clusters_calib = self.calibrate_cluster(marker_names, markers_data)
        self.refine_calibration()

    def calibrate_cluster(self, marker_names, markers_data):
        calibration_dict = {}
        for cluster_name, cluster_markers in self.options.clusters.items():
            if all(marker in marker_names for marker in cluster_markers[0]):
                rt = get_local_cs(markers_data, cluster_markers[0])
                for marker_local_name in cluster_markers[1]:
                    marker_in_local = np.dot(np.linalg.inv(rt[..., 0]), np.concatenate((markers_data[:, marker_names.index(marker_local_name), 0], [1])))
                    calibration_dict[marker_local_name] = marker_in_local
        self.calibration_data.update(calibration_dict)
        return calibration_dict

    def apply_cluster_calibration(self, data, cluster_name, calibration_dict):
        pass
    def init_thorax_calibration(self, thorax_file_path=None):
        if thorax_file_path is not None:
            self.thorax_file_path = thorax_file_path
        self.thorax_calibration_data = ezc3d.c3d(self.thorax_file_path)
        markers_names = list(self.thorax_calibration_data["parameters"]["POINT"]["LABELS"]["value"])
    
    def init_arms_calibration(self, arms_file_path=None):
        if arms_file_path is not None:
            self.arms_file_path = arms_file_path
        self.anato_calibration_data = ezc3d.c3d(self.arms_file_path)
        markers_names = list(self.anato_calibration_data["parameters"]["POINT"]["LABELS"]["value"])
        # Placeholder for arms calibration initialization logic
        pass

    def _from_thorax(self, thorax_data):
        # Placeholder for processing thorax data
        pass
    
    def _from_arms(self, arms_data):
        # Placeholder for processing arms data
        pass

    @staticmethod
    def get_events(events, data_rate=100):
        if "LABELS" not in events:
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
        return paired_event_idx.astype(int)