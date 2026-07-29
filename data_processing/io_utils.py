import numpy as np
from C3DtoTRC import WriteTrcFromMarkersData
import os
import opensim as osim


def write_mot_file(filename, time_step, dof_names, states):
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    [labels.append(name) for name in dof_names]
    table.setColumnLabels(labels)
    for row in range(states.shape[1]):
        table.appendRow(time_step * row, osim.RowVector(states[:, row]))
    if os.path.exists(filename):
        os.remove(filename)
    osim.STOFileAdapter.write(table, filename)


def write_trc(data, names, output_file_path, data_rate):
    "data: 3xmxn"
    WriteTrcFromMarkersData(
        output_file_path=output_file_path,
        markers=data,
        marker_names=names,
        data_rate=data_rate,
        cam_rate=data_rate,
        n_frames=data.shape[2],
        start_frame=1,
        units="m",
    ).write()

def return_unique_keypoints(data_path):
    "Return unique keypoints from data_path"
    keypoints = np.load(data_path)
    idx = keypoints[:, 0, 3]
    _, i = np.unique(idx, return_index=True)
    keypoints = keypoints[i, :, :3]
    return keypoints, idx[i].astype(int)