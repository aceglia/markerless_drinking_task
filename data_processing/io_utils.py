import numpy as np
from C3DtoTRC import WriteTrcFromMarkersData
import os
import opensim as osim


# def write_mot_file(filename, time_step, dof_names, states):
#     table = osim.TimeSeriesTable()
#     labels = osim.StdVectorString()
#     [labels.append(name) for name in dof_names]
#     table.setColumnLabels(labels)
#     for row in range(states.shape[1]):
#         table.appendRow(time_step * row, osim.RowVector(states[:, row]))
#     if os.path.exists(filename):
#         os.remove(filename)
#     osim.STOFileAdapter.write(table, filename)


def write_mot_file(filename, time_step, dof_names, states):
    states = np.asarray(states)

    if states.ndim != 2:
        raise ValueError(f"states must be 2D, got shape {states.shape}")

    n_dofs, n_frames = states.shape

    if len(dof_names) != n_dofs:
        raise ValueError(
            f"Number of DOF names ({len(dof_names)}) "
            f"does not match states ({n_dofs})"
        )

    time = np.arange(n_frames) * time_step

    with open(filename, "w", newline="\n") as f:

        # MOT header
        f.write("Coordinates\n")
        f.write("version=1\n")
        f.write(f"nRows={n_frames}\n")
        f.write(f"nColumns={n_dofs + 1}\n")
        f.write("inDegrees=no\n")
        f.write("endheader\n")

        # Column names
        f.write("time\t" + "\t".join(str(name) for name in dof_names) + "\n")

        # Data
        for i in range(n_frames):
            values = [time[i]] + states[:, i].tolist()

            f.write(
                "\t".join(f"{value:.10g}" for value in values)
                + "\n"
            )


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