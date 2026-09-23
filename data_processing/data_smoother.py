import numpy as np
from filterpy.kalman import KalmanFilter


class MarkerKalmanSmoother:
    """
    6D constant-velocity Kalman filter for a 3D marker.

    State:
        [x, y, z, vx, vy, vz]

    Measurement:
        [x, y, z]
    """

    def __init__(
        self,
        dt,
        process_noise=1.0,
        measurement_noise=1e-4,
        initial_position=None,
        initial_velocity=None,
    ):
        self.dt = dt

        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # ---------------------------------------------------------
        # State transition
        # ---------------------------------------------------------
        self.kf.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0, dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=float)

        # ---------------------------------------------------------
        # Measurement model
        # ---------------------------------------------------------
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # ---------------------------------------------------------
        # Process noise
        #
        # White-noise acceleration model
        # ---------------------------------------------------------
        q = process_noise

        dt = self.dt

        self.kf.Q = q * np.array([
            [dt**4 / 4, 0,          0,          dt**3 / 2, 0,          0],
            [0,          dt**4 / 4, 0,          0,          dt**3 / 2, 0],
            [0,          0,          dt**4 / 4, 0,          0,          dt**3 / 2],

            [dt**3 / 2, 0,          0,          dt**2,     0,          0],
            [0,          dt**3 / 2, 0,          0,          dt**2,     0],
            [0,          0,          dt**3 / 2, 0,          0,          dt**2],
        ])

        # ---------------------------------------------------------
        # Measurement noise
        # ---------------------------------------------------------
        self.kf.R = measurement_noise * np.eye(3)

        # ---------------------------------------------------------
        # Initial covariance
        # ---------------------------------------------------------
        self.kf.P = np.diag([
            1e-4,  # x
            1e-4,  # y
            1e-4,  # z
            1e-2,  # vx
            1e-2,  # vy
            1e-2,  # vz
        ])

        # ---------------------------------------------------------
        # Initial state
        # ---------------------------------------------------------
        if initial_position is None:
            initial_position = np.zeros(3)

        if initial_velocity is None:
            initial_velocity = np.zeros(3)

        self.kf.x = np.hstack([
            initial_position,
            initial_velocity
        ]).reshape(6, 1)

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        return self.kf.x[:3, 0].copy()

    def innovation(self, measurement):
        """
        Calculate innovation and normalized innovation squared (NIS)
        without updating the filter.
        """

        z = np.asarray(measurement).reshape(3, 1)

        y = z - self.kf.H @ self.kf.x

        S = (
            self.kf.H @ self.kf.P @ self.kf.H.T
            + self.kf.R
        )

        nis = float(
            (y.T @ np.linalg.inv(S) @ y).item()
        )

        return y[:, 0], S, nis

    def update(self, measurement):
        """Update using a valid measurement."""
        z = np.asarray(measurement).reshape(3, 1)
        self.kf.update(z)

    def step(self, measurement, nis_threshold=None):
        """
        Predict and optionally update.

        If nis_threshold is provided and the measurement is an
        outlier, the measurement is rejected.
        """

        self.predict()
        innovation = np.nan
        S = np.nan
        nis = np.nan
        accepted = False
        
        if np.isfinite(measurement).all():
            innovation, S, nis = self.innovation(measurement)

            accepted = True

            if nis_threshold is not None and nis > nis_threshold:
                accepted = False
            else:
                self.update(measurement)

        return {
            "position": self.kf.x[:3, 0].copy(),
            "velocity": self.kf.x[3:, 0].copy(),
            "innovation": innovation,
            "nis": nis,
            "accepted": accepted,
        }

def smooth_marker(
    trajectory,
    dt,
    process_noise=1.0,
    measurement_noise=1e-4,
    nis_threshold=None,
):
    """
    trajectory:
        (N, 3) array containing [x, y, z]

    Returns:
        smoothed_position : (N, 3)
        smoothed_velocity : (N, 3)
        nis               : (N,)
        accepted          : (N,)
    """

    trajectory = np.asarray(trajectory, dtype=float)

    n = len(trajectory)

    if trajectory.shape[1] != 3:
        raise ValueError(
            "trajectory must have shape (N, 3)"
        )

    # ---------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------

    kf = MarkerKalmanSmoother(
        dt=dt,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_position=trajectory[0],
    )

    # Store filtered states
    means = np.zeros((n, 6))
    covariances = np.zeros((n, 6, 6))

    nis_values = np.zeros(n)
    accepted = np.ones(n, dtype=bool)

    # First frame
    means[0] = kf.kf.x[:, 0]
    covariances[0] = kf.kf.P

    # ---------------------------------------------------------
    # Forward Kalman filter
    # ---------------------------------------------------------

    for i in range(1, n):

        result = kf.step(
            trajectory[i],
            nis_threshold=nis_threshold,
        )

        means[i] = kf.kf.x[:, 0]
        covariances[i] = kf.kf.P

        nis_values[i] = result["nis"]
        accepted[i] = result["accepted"]

    # ---------------------------------------------------------
    # RTS smoother
    # ---------------------------------------------------------

    xs, Ps, Ks, _ = kf.kf.rts_smoother(
        means,
        covariances,
    )

    smoothed_position = xs[:, :3]
    smoothed_velocity = xs[:, 3:]

    return (
        smoothed_position,
        smoothed_velocity,
        nis_values,
        accepted,
    )