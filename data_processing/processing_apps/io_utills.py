import ezc3d
import os
import numpy as np

class ViconData:
    def __init__(self, c3d_file_path):
        self.c3d_file_path = c3d_file_path
        self.data = self.load_c3d_file()

    def load_c3d_file(self):
        if not os.path.exists(self.c3d_file_path):
            raise FileNotFoundError(f"C3D file not found: {self.c3d_file_path}")
        return ezc3d.c3d(self.c3d_file_path)

    def get_marker_data(self):
        return self.data['data']['points']

    def get_analog_data(self):
        return self.data['data']['analogs']

    def get_frame_rate(self):
        return self.data['header']['frame_rate']

    def get_number_of_frames(self):
        return self.data['header']['points']['n_frames']