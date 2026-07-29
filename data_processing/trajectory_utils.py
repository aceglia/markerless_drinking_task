import numpy as np
from sklearn.cluster import DBSCAN
from scipy import signal

def filter_points_3d(data, cutoff=10, fs=30, order=2):
    b, a = signal.butter(order, cutoff, "low", fs=fs)
    filtered_mat = np.zeros_like(data)
    t = np.arange(data.shape[0])
    data_interp = np.copy(data)
    for i in range(data.shape[1]):
        for j in range(3):
            valid = ~np.isnan(data_interp[:, i, j])
            len_val = valid.nonzero()[0].shape[0]
            if len_val == data_interp.shape[0]:
                continue
            data_interp[:, i, j] = np.interp(t, t[valid], data_interp[valid, i, j])
        filtered_sig = signal.filtfilt(b, a, data_interp[:, i].T)
        filtered_mat[:, i, :] = filtered_sig.T
    return filtered_mat


def remove_outliers(keypoints, on_diff=True):
    filtered = np.zeros_like(keypoints) * np.nan
    if on_diff:
        filtered[-1] = keypoints[-1]
    to_eval = keypoints
    if on_diff:
        to_eval = keypoints[1:] - keypoints[:-1]
    for ch in range(keypoints.shape[1]):
        ch_data = abs(to_eval[:, ch, :])
        mean = np.nanmean(ch_data[ch_data[:, 2].nonzero()[0], :], axis=0)
        std = np.nanstd(ch_data[ch_data[:, 2].nonzero()[0], :], axis=0)

        lower = mean - 3 * std
        upper = mean + 3 * std
        valid_mask = np.all((ch_data >= lower) & (ch_data <= upper), axis=1)
        if on_diff:
            filtered[:-1, ch, :][valid_mask] = keypoints[:-1, ch, :][valid_mask]
        else:
            filtered[:, ch, :][valid_mask] = keypoints[:, ch, :][valid_mask]
    return filtered


def fill_gaps(traj):
    traj_filled = np.copy(traj)
    for i in range(traj.shape[1]):
        valid = ~np.isnan(traj[:, i])
        if valid.sum() == 0:
            continue
        traj_filled[:, i] = np.interp(np.arange(traj.shape[0]), np.arange(traj.shape[0])[valid], traj[valid, i])
    return traj_filled


def clean_trajectory(keypoints, idxs=None, eps_list=[0.1, 0.08, 0.06, 0.04, 0.02, 0.01]):
    filtered = np.copy(keypoints)
    filtered[filtered == 0] = np.nan
    if idxs is None:
        idxs = list(range(keypoints.shape[1]))
    for ch in idxs:
        for i in range(len(eps_list)):
            eps = eps_list[i]
            ch_diff = abs(filtered[1:, ch, -1] - filtered[:-1, ch, -1])
            ch_diff_x = abs(filtered[1:, ch, 0] - filtered[:-1, ch, 0])
            ch_diff_y = abs(filtered[1:, ch, 1] - filtered[:-1, ch, 1])
            max_ch_x = np.nanmax(ch_diff_x)
            max_ch_y = np.nanmax(ch_diff_y)
            max_ch = np.nanmax(ch_diff)
            ratio = max_ch / np.min([max_ch_x, max_ch_y]) + 1e-6
            if ratio > 1:
                nan_mask = np.isnan(filtered[:, ch, :]).any(axis=1)
                valid_data = filtered[:, ch, :][~nan_mask]
                clustering = DBSCAN(eps=eps, min_samples=5).fit(valid_data)
                labels = clustering.labels_
                valid_labels = labels[labels >= 0]
                if len(valid_labels):
                    main_label = np.bincount(valid_labels).argmax()
                    valid_data[labels != main_label] = np.nan

                data_rebuilt = np.full_like(filtered[:, ch, :], np.nan)
                data_rebuilt[~nan_mask] = valid_data

                filtered[:, ch, :] = data_rebuilt
            else:
                break
        if not np.all(np.isfinite(filtered[:, ch, :])):
            filtered[:, ch, :] = fill_gaps(filtered[:, ch, :])
    return filtered