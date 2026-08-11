import os
import pickle

import numpy as np
import opensim as osim
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from .io_utils import write_mot_file

from enum import IntEnum


class MarkerNoise(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class BoundedMerweSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, n, theta_bounds, **kwargs):
        super().__init__(n=n, **kwargs)
        self.theta_bounds = theta_bounds

    def sigma_points(self, x, P):
        sigmas = super().sigma_points(x, P)
        for i in range(sigmas.shape[0]):
            for j in range(len(self.theta_bounds)):
                min_j, max_j = self.theta_bounds[j]
                sigmas[i, j] = np.clip(sigmas[i, j], min_j, max_j)
        return sigmas


class JointMarkerUKF:
    def __init__(
        self,
        model,
        data_rate=30,
        with_markers=False,
        type="constant_velocity",
        experimental_marker_names=None,
        model_marker_names=None,
    ):
        self.model = model if isinstance(model, osim.Model) else osim.Model(model)
        self.state = self.model.initSystem()

        self.dt = 1.0 / data_rate
        if type == "constant_velocity":
            self.n_diff = 1
        elif type == "constant_acceleration":
            self.n_diff = 2
        else:
            raise RuntimeError("Type is not a valid type")
        self.with_markers = with_markers
        self.dof_names = tuple(s.toString() for s in self.model.getCoordinateSet())
        self.coordinates = [self.model.getCoordinateSet().get(coord) for coord in self.dof_names]
        self.markers = [self.model.getMarkerSet().get(i) for i in range(self.model.getMarkerSet().getSize())]
        self.model_marker_names = [str(marker.getName()) for marker in self.markers]
        self.locked_idxs = [
            i
            for i in range(self.model.getCoordinateSet().getSize())
            if self.model.getCoordinateSet().get(i).get_locked()
        ]
        self.marker_order = None
        self.initial_states = self.model.getStateVariableValues(self.state).to_numpy()

        self.N_JOINTS = self.model.getCoordinateSet().getSize()
        self.N_MARKERS = self.model.getMarkerSet().getSize()

        self.N_ACTIVE_JOINTS = self.N_JOINTS - len(self.locked_idxs)

        self.joint_mins = np.array(
            [
                self.model.getCoordinateSet().get(i).getRangeMin()
                for i in range(self.N_JOINTS)
                if i not in self.locked_idxs
            ]
        )
        self.joint_maxs = np.array(
            [
                self.model.getCoordinateSet().get(i).getRangeMax()
                for i in range(self.N_JOINTS)
                if i not in self.locked_idxs
            ]
        )

        self.theta_bounds = list(zip(self.joint_mins, self.joint_maxs))

        self.dim_x = (self.n_diff + 1) * self.N_ACTIVE_JOINTS
        if self.with_markers:
            self.dim_x += 3 * self.N_MARKERS * 2

        self.dim_z = 3 * self.N_MARKERS

        self.points = BoundedMerweSigmaPoints(
            n=self.dim_x, alpha=0.5, beta=2.0, kappa=0.0, theta_bounds=self.theta_bounds
        )
        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=self.dt, fx=self.fx, hx=self.hx, points=self.points)

        self.ukf.x = np.zeros(self.dim_x)
        self.experimental_marker_names = experimental_marker_names
        # self._init_ukf(first_marker_frame, update_q=True)
        if self.experimental_marker_names is not None:
            self.set_marker_order(self.experimental_marker_names, model_marker_names=model_marker_names)

    def set_marker_order(self, experimental_marker_names, model_marker_names=None):
        if model_marker_names is None:
            model_marker_names = self.model_marker_names
        if len(model_marker_names) != self.N_MARKERS:
            raise ValueError(f"model_marker_names must have {self.N_MARKERS} entries, got {len(model_marker_names)}.")
        if len(set(experimental_marker_names)) != len(experimental_marker_names):
            raise ValueError("experimental_marker_names must contain unique entries.")

        exp_index = {name: i for i, name in enumerate(experimental_marker_names)}
        missing = [name for name in model_marker_names if name not in exp_index]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing markers in experimental data: {missing_str}")

        self.marker_order = np.array([exp_index[name] for name in model_marker_names], dtype=int)

    def _coerce_marker_frame(self, marker_frame):
        marker_frame = np.asarray(marker_frame)
        if marker_frame.ndim == 1:
            if marker_frame.size % 3 != 0:
                raise ValueError("Marker frame must have a length multiple of 3.")
            marker_frame = marker_frame.reshape(3, -1)
        elif marker_frame.ndim == 2:
            if marker_frame.shape[0] == 3:
                pass
            elif marker_frame.shape[1] == 3:
                marker_frame = marker_frame.T
            else:
                raise ValueError("Marker frame must be shaped (3, N) or (N, 3).")
        else:
            raise ValueError("Marker frame must be 1D or 2D.")
        return marker_frame

    def _coerce_marker_sequence(self, markers, exp_marker_names=None):
        if exp_marker_names is not None:
            self.set_marker_order(exp_marker_names)

        markers = np.asarray(markers)
        if markers.ndim != 3:
            raise ValueError("Markers must be a 3D array.")
        if markers.shape[0] == 3:
            return markers
        if markers.shape[2] == 3:
            if markers.shape[1] == self.N_MARKERS:
                return markers.transpose(2, 1, 0)
            if markers.shape[0] == self.N_MARKERS:
                return markers.transpose(2, 0, 1)
            raise ValueError("Markers with a trailing 3-axis must include the model marker count.")
        if markers.shape[1] == 3:
            if markers.shape[0] == self.N_MARKERS:
                return markers.transpose(1, 0, 2)
            if markers.shape[2] == self.N_MARKERS:
                return markers.transpose(1, 2, 0)
            raise ValueError("Markers with a middle 3-axis must include the model marker count.")
        raise ValueError("Markers must contain a 3-length axis for XYZ coordinates.")

    def _reorder_markers(self, markers):
        if self.marker_order is None:
            return markers
        if markers.ndim == 2:
            if markers.shape[1] <= np.max(self.marker_order):
                raise ValueError("Marker frame does not match experimental marker ordering.")
            return markers[:, self.marker_order]
        if markers.ndim == 3:
            if markers.shape[1] <= np.max(self.marker_order):
                raise ValueError("Marker sequence does not match experimental marker ordering.")
            return markers[:, self.marker_order, :]
        raise ValueError("Markers must be 2D or 3D for reordering.")

    def _prepare_marker_frame(self, marker_frame):
        marker_frame = self._coerce_marker_frame(marker_frame)
        marker_frame = self._reorder_markers(marker_frame)
        if marker_frame.shape[1] != self.N_MARKERS:
            raise ValueError(f"Expected {self.N_MARKERS} markers, got {marker_frame.shape[1]}.")
        self._update_marker_validity(marker_frame)
        return marker_frame.flatten()

    def _update_marker_validity(self, marker_frame):
        """Mark which markers have valid data and adjust measurement noise accordingly."""
        self.marker_valid = np.ones(self.N_MARKERS, dtype=bool)
        for i in range(self.N_MARKERS):
            if np.any(np.isnan(marker_frame[:, i])):
                self.marker_valid[i] = False
        self._update_measurement_noise()

    def _update_measurement_noise(self):
        """Set per-marker measurement noise: high for invalid markers, low for valid ones."""
        R_diag = np.ones(self.dim_z)
        for i in range(self.N_MARKERS):
            if not self.marker_valid[i]:
                R_diag[3 * i : 3 * i + 3] = 1e6
            elif not hasattr(self, "marker_noise_base"):
                R_diag[3 * i : 3 * i + 3] = self.ukf.R[3 * i, 3 * i]
        self.ukf.R = np.diag(R_diag)
    
    def _augment_q(self, q):
        mask_idx = [True if i not in self.locked_idxs else False for i in range(self.N_JOINTS)]
        q_full = np.zeros(self.N_JOINTS)
        dq_full = None if q.shape[0] == self.N_ACTIVE_JOINTS else np.zeros(self.N_JOINTS)
        _q = q if q.shape[0] == self.N_ACTIVE_JOINTS else q[::2]
        _dq = None if q.shape[0] == self.N_ACTIVE_JOINTS else q[1::2]
        q_full[mask_idx] = _q
        if _dq is not None:
            dq_full[mask_idx] = _dq
            q_tot = np.array([[q, dq] for q, dq in zip(q_full, dq_full)]).flatten()
            return q_tot
        return q_full

    def set_joint_angles(self, theta):
        map_q = np.array([[q, 0] for q in theta]).flatten()
        map_q = self._augment_q(map_q)
        self.initial_states[: self.N_JOINTS * 2] = map_q
        self.model.setStateVariableValues(self.state, osim.Vector(self.initial_states))
        self.model.realizePosition(self.state)
        # self.model.assemble(self.state)

    def _get_transition_matrix(self):
        A = np.eye(self.dim_x)
        num_joints, num_diff, time_step, dim_z = self.N_ACTIVE_JOINTS, self.n_diff, self.dt, self.dim_z
        with_markers = self.with_markers

        for i in range(num_joints):
            A[i, num_joints + i] = time_step
            if num_diff == 2:
                A[i, num_diff * num_joints + i] = 0.5 * time_step**2
                A[num_joints + i, (num_joints * num_diff) + i] = time_step

        if with_markers:
            start_index_markers = num_joints * (num_diff + 1)
            for i in range(dim_z):
                A[start_index_markers + i, start_index_markers + dim_z + i] = time_step

        return A

    def fx(self, x, dt):
        x_new = self.transition_matrix @ x
        x_new[: self.N_ACTIVE_JOINTS] = np.clip(x_new[: self.N_ACTIVE_JOINTS], self.joint_mins, self.joint_maxs)
        return x_new

    def hx(self, x):
        theta = x[: self.N_ACTIVE_JOINTS]

        self.set_joint_angles(theta)
        markers_pos = np.zeros((3, self.N_MARKERS))
        for i in range(self.N_MARKERS):
            pos = self.markers[i].getLocationInGround(self.state)
            markers_pos[:, i] = [pos.get(j) for j in range(3)]

        return markers_pos.flatten()

    def initialize(self, first_marker_frame, marker_noise_lvl: MarkerNoise = MarkerNoise.NONE):
        self.marker_noise_lvl = marker_noise_lvl
        self.marker_valid = np.ones(self.N_MARKERS, dtype=bool)
        self._get_kalman_matrix()
        prepared_frame = self._prepare_marker_frame(first_marker_frame)

        initial_q = self._fast_ik(prepared_frame)
        self.set_joint_angles(initial_q)

        self._init_ukf(prepared_frame)

    def _get_kalman_matrix(self):
        self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-2
        self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-3
        self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-4
        # if self.marker_noise_lvl == MarkerNoise.NONE:
        #     self.marker_noise_base = 1e-6
        #     self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-5
        #     self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-6
        #     self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-6
        # elif self.marker_noise_lvl == MarkerNoise.LOW:
        #     self.marker_noise_base = 1e-4
        #     self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-3
        #     self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-4
        #     self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-4
        # elif self.marker_noise_lvl == MarkerNoise.MEDIUM:
        #     self.marker_noise_base = 1e-3
        #     self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-2
        #     self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-3
        #     self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-3
        # elif self.marker_noise_lvl == MarkerNoise.HIGH:
        #     self.marker_noise_base = 1e-2
        #     self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-1
        #     self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-2
        #     self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-2
        # else:
        #     raise RuntimeError("Marker noise level not recognized")
    def marker_positions(self, q, realize_position=True):
        if realize_position:
            self.set_joint_angles(q)
        markers_model = np.zeros((3, self.N_MARKERS))

        for i in range(self.N_MARKERS):
            pos = self.markers[i].getLocationInGround(self.state)
            markers_model[:, i] = [
                pos.get(0),
                pos.get(1),
                pos.get(2),
            ]

        return markers_model

    def _opensim_ik(self, marker_frame):
        """
        Single-frame OpenSim IK initialization.

        Parameters
        ----------
        marker_frame : ndarray
            Flattened marker vector (3*N_MARKERS,)
            in model marker order.

        Returns
        -------
        q : ndarray
            Coordinate values.
        """

        marker_array = marker_frame.reshape(3, -1)
        marker_names = osim.StdVectorString()
        marker_weights = osim.SetMarkerWeights()

        # Liste temporaire en Python pour stocker uniquement les coordonnées valides
        valid_coords = []

        # 2. Filtrer et extraire uniquement les marqueurs non-NaN
        for i in range(self.N_MARKERS):
            name = self.markers[i].getName()

            # Si le marqueur contient un NaN, on l'ignore complètement
            if np.any(np.isnan(marker_array[:, i])):
                continue

            marker_names.append(name)

            weight = osim.MarkerWeight(name, 1.0)
            marker_weights.cloneAndAppend(weight)

            # Ajouter les coordonnées valides à notre liste temporaire
            valid_coords.append(
                osim.Vec3(
                    float(marker_array[0, i]),
                    float(marker_array[1, i]),
                    float(marker_array[2, i]),
                )
            )

        valid_count = len(valid_coords)

        if valid_count == 0:
            raise RuntimeError("No valid markers available.")

        # 3. Construire le RowVectorVec3 avec la TAILLE EXACTE des marqueurs valides
        row = osim.RowVectorVec3(valid_count)
        # for idx, vec_xyz in enumerate(valid_coords):
        #    row[idx] = vec_xyz
        for dynamic_idx, static_idx in enumerate(valid_coords):
            v = osim.Vec3(
                float(marker_array[0, static_idx]),
                float(marker_array[1, static_idx]),
                float(marker_array[2, static_idx]),
            )
            # Utiliser la méthode d'accès native C++ explicite si l'opérateur [] échoue
            row.set(dynamic_idx, v)

        # 4. Assigner à la table de données d'OpenSim
        marker_locations = osim.TimeSeriesTableVec3()
        marker_locations.setColumnLabels(marker_names)
        marker_locations.appendRow(0.0, row)

        # 5. Instancier la référence pour le solveur IK
        markers_ref = osim.MarkersReference(
            marker_locations,
            marker_weights,
        )

        # ---------------------------------------
        # Empty coordinate references
        # ---------------------------------------

        # coord_refs = osim.ArrayCoordinateReference()
        coord_refs = osim.SimTKArrayCoordinateReference()

        solver = osim.InverseKinematicsSolver(
            self.model,
            markers_ref,
            coord_refs,
        )

        # ---------------------------------------
        # Initial pose
        # ---------------------------------------

        # state = self.state
        state = self.model.initSystem()  # Forcer la ré-allocation propre du vecteur d'état
        self.model.realizeTopology(state)

        for i in range(self.N_JOINTS):
            self.coordinates[i].setValue(state, self.coordinates[i].getDefaultValue())

        self.model.realizePosition(state)
        self.model.realizeVelocity(state)
        print(state.getY().size())
        print(state.getY().get(0))

        # ---------------------------------------
        # Solve
        # ---------------------------------------
        # 4. Step through time frames programmatically
        time_series = markers_ref.getValidTimeRange()
        start_time = time_series.get(0)
        end_time = time_series.get(1)

        # Loop over structural data frames inside the session memory
        current_time = start_time
        state.setTime(current_time)
        solver.assemble(state)

        q = np.array([self.coordinates[i].getValue(state) for i in range(self.N_JOINTS)])

        # ---------------------------------------
        # Diagnostics
        # ---------------------------------------

        errs = osim.ArrayDouble()
        solver.computeCurrentMarkerErrors(errs)

        rmse = np.sqrt(np.mean([errs.get(j) ** 2 for j in range(errs.getSize())]))

        print(f"OpenSim IK RMSE: {rmse*1000:.2f} mm")

        return q

    def _fast_ik(self, marker_frame):
        """Initialization IK used once before UKF startup."""
        from scipy import optimize

        marker_array = marker_frame.reshape(3, -1)

        valid_idxs = ~np.isnan(marker_array[0, :])
        valid_marker_indices = np.where(valid_idxs)[0]

        if len(valid_marker_indices) == 0:
            raise RuntimeError("No valid markers available for IK initialization")

        markers_valid = marker_array[:, valid_marker_indices]
        # --------------------------------------------------
        # Initial guess
        # --------------------------------------------------

        try:
            x0 = np.array([self.coordinates[i].getDefaultValue() for i in range(self.N_JOINTS)])
        except Exception:
            x0 = (self.joint_mins + self.joint_maxs) / 2

        x0 = np.delete(x0, self.locked_idxs)
        x0 = np.clip(x0, self.joint_mins, self.joint_maxs)

        # --------------------------------------------------
        # Residual
        # --------------------------------------------------

        def marker_diff(q):
            markers_model = self.marker_positions(q)
            markers_model_valid = markers_model[:, valid_marker_indices]

            return (markers_model_valid - markers_valid).ravel(order="F")

        # --------------------------------------------------
        # Central-difference Jacobian
        # --------------------------------------------------

        def marker_jacobian(q, eps=1e-3):

            n_res = 3 * len(valid_marker_indices)
            J = np.zeros((n_res, self.N_ACTIVE_JOINTS))
            count = 0
            for j in range(self.N_JOINTS):
                if j in self.locked_idxs:
                    continue
                q_plus = q.copy()
                q_minus = q.copy()

                q_plus[count] += eps
                q_minus[count] -= eps

                f_plus = self.marker_positions(q_plus)[:, valid_marker_indices].ravel(order="F")

                f_minus = self.marker_positions(q_minus)[:, valid_marker_indices].ravel(order="F")

                J[:, count] = (f_plus - f_minus) / (2 * eps)
                count += 1

            return J

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------

        sol = optimize.least_squares(
            fun=marker_diff,
            x0=x0,
            jac=marker_jacobian,
            bounds=(self.joint_mins, self.joint_maxs),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=500,
            verbose=0,
        )

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        residual = marker_diff(sol.x)

        rmse = np.sqrt(np.mean(residual.reshape(-1, 3) ** 2))
        # euclid_dist = np.linalg.norm(residual.reshape(-1, 3), axis=1)
        # max_euclid_dist = np.max(euclid_dist)
        # marker_max = np.where(euclid_dist == max_euclid_dist)

        print(f"IK init: success={sol.success}, " f"RMSE={rmse:.4f} m, " f"nfev={sol.nfev}")

        return sol.x

    def set_custom_matrix(self, P, Q, R):
        self.ukf.P = P
        self.ukf.Q = Q
        self.ukf.R = R

    def _init_ukf(self, first_marker_frame, update_q=True):
        self.transition_matrix = self._get_transition_matrix()
        get_state = self.model.getStateVariableValues(self.state).to_numpy()[: self.N_JOINTS * 2][::2]

        self.ukf.x[: self.N_ACTIVE_JOINTS] = get_state[[i for i in range(self.N_JOINTS) if i not in self.locked_idxs]]
        self.ukf.x[self.N_ACTIVE_JOINTS : self.N_ACTIVE_JOINTS * (self.n_diff + 1)] = 0
        if self.with_markers:
            start_index_markers = self.N_ACTIVE_JOINTS * (self.n_diff + 1)
            self.ukf.x[start_index_markers : start_index_markers + self.dim_z] = first_marker_frame

    def step(self, marker_frame):
        marker_frame = self._prepare_marker_frame(marker_frame)

        marker_array = marker_frame.reshape(3, -1)
        marker_array_flat = marker_array.flatten()

        for i in range(self.N_MARKERS):
            if not self.marker_valid[i]:
                marker_array_flat[3 * i : 3 * i + 3] = np.nan

        self.ukf.predict()

        valid_idxs = ~np.isnan(marker_array_flat)
        if np.any(valid_idxs):
            z_valid = marker_array_flat[valid_idxs]
            self.ukf.update(z_valid, hx=lambda x: self.hx(x)[valid_idxs])

        theta_est = self.ukf.x[: self.N_ACTIVE_JOINTS]
        if self.with_markers:
            start_index_markers = self.N_ACTIVE_JOINTS * (self.n_diff + 1)
            marker_est = self.ukf.x[start_index_markers : start_index_markers + self.dim_z].reshape(3, -1)
            return theta_est, marker_est
        return self.ukf.x

    def run(self, markers):
        self.expe_markers = self._coerce_marker_sequence(markers)
        self.states = np.empty((self.N_JOINTS * 2, self.expe_markers.shape[-1]))
        self.model_markers = np.zeros((self.expe_markers.shape[0], self.N_MARKERS, self.expe_markers.shape[-1]))
        self.initialize(self.expe_markers[:, :, 0])
        for i in range(self.expe_markers.shape[-1]):
            states_tmp = self.step(self.expe_markers[:, :, i])
            self.model_markers[... , i] = self.marker_positions(states_tmp[:self.N_ACTIVE_JOINTS], realize_position=False)
            states_tmp = states_tmp[:self.N_ACTIVE_JOINTS * 2]
            self.states[:self.N_JOINTS, i] = self._augment_q(states_tmp[:self.N_ACTIVE_JOINTS])
            self.states[self.N_JOINTS :, i] = self._augment_q(states_tmp[self.N_ACTIVE_JOINTS :])
        return self.states

    def save(self, filename, initial_dir = None, mot_file=False):
        dic_to_save = {
            'model_path': self.model.getDocumentFileName(),
            'dt': self.dt,
            'n_diff': self.n_diff,
            'with_markers': self.with_markers,
            'experimental_marker_names': self.experimental_marker_names,
            'model_marker_names': self.model_marker_names,
            'locked_idxs': self.locked_idxs,
            'joint_mins': self.joint_mins,
            'joint_maxs': self.joint_maxs,
            'experimental_markers': self.expe_markers,
            'model_markers': self.model_markers,
            'q': self.states[:self.N_JOINTS],
            'dq': self.states[self.N_JOINTS :],
            'dof_names': self.dof_names,
        }
        if initial_dir is not None:
            dic_to_save.update(initial_dir)
        with open(filename, 'wb') as f:
            pickle.dump(dic_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        if mot_file:
            write_mot_file(filename.replace(".pkl", '.mot'), self.dt, self.dof_names, self.states[:self.N_JOINTS])       
