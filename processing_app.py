import sys
from data_processing.vicon_processing.app import ViconProcessingApp
from PyQt5.QtWidgets import QApplication
import ezc3d

if __name__ == "__main__":
    c3d_file = r"C:\Users\neuromobility_lab\Documents\CIME_MS\test_002\vicon\DrinkingLeftArm.c3d"
    c3d_data = ezc3d.c3d(c3d_file)
    marker_names_file = c3d_data["parameters"]["POINT"]["LABELS"]["value"]

    app = QApplication(sys.argv)
    window = ViconProcessingApp()
    window.show()
    sys.exit(app.exec_())