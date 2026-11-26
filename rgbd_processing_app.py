import torch
import sys
from data_processing.processing_apps.rgbd_app import RGBDProcessingApp
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RGBDProcessingApp()
    window.show() 
    sys.exit(app.exec())