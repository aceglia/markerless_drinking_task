import torch
import sys
from data_processing.processing_apps.vicon_app import ViconProcessingApp
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViconProcessingApp()
    window.show()
    sys.exit(app.exec())