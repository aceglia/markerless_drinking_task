import csv
import json
import os
import pickle
import shutil
import time
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QGridLayout,
)
from .process_utils import ViconProcessor
import numpy as np


class ViconProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vicon Processing App")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._init_layout()
        self.processor = None

    def _init_layout(self):
        # Create file selection layout
        file_selection_layout = QGridLayout()
        self.thorax_file_label = QLabel("Thorax calibrationFile:")
        self.thorax_file_input = QLineEdit()
        self.thorax_file_browse_button = QPushButton("Browse")
        self.thorax_file_browse_button.clicked.connect(self.browse_thorax_file)

        self.arms_file_label = QLabel("Anatomical File:")
        self.arms_file_input = QLineEdit()
        self.arms_file_browse_button = QPushButton("Browse")
        self.arms_file_browse_button.clicked.connect(self.browse_arms_file)

        file_selection_layout.addWidget(self.thorax_file_label, 0, 0)
        file_selection_layout.addWidget(self.thorax_file_input, 0, 1)
        file_selection_layout.addWidget(self.thorax_file_browse_button, 0, 2)
        file_selection_layout.addWidget(self.arms_file_label, 1, 0)
        file_selection_layout.addWidget(self.arms_file_input, 1, 1)
        file_selection_layout.addWidget(self.arms_file_browse_button, 1, 2)

        self.trials_file_label = QLabel("Trials File:")
        self.trials_file_input = QLineEdit()
        self.trials_file_browse_button = QPushButton("Browse")
        self.trials_file_browse_button.clicked.connect(self.browse_trials_file)

        file_selection_layout.addWidget(self.trials_file_label, 2, 0)
        file_selection_layout.addWidget(self.trials_file_input, 2, 1)
        file_selection_layout.addWidget(self.trials_file_browse_button, 2, 2)

        self.opensim_file_label = QLabel("OpenSim File:")
        self.opensim_file_input = QLineEdit()
        self.opensim_file_browse_button = QPushButton("Browse")
        self.opensim_file_browse_button.clicked.connect(self.browse_opensim_file)
        self.opensim_scale_button = QPushButton("Scale")
        self.opensim_scale_button.clicked.connect(self.scale_opensim_model)

        file_selection_layout.addWidget(self.opensim_file_label, 3, 0)
        file_selection_layout.addWidget(self.opensim_file_input, 3, 1)
        file_selection_layout.addWidget(self.opensim_file_browse_button, 3, 2)
        file_selection_layout.addWidget(self.opensim_scale_button, 3, 3)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)

        file_selection_layout.addWidget(self.process_button, 4, 0, 1, 4)

        self.central_widget.setLayout(file_selection_layout)

    def browse_thorax_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Thorax Calibration File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.thorax_file_input.setText(file_path)
            self.check_files_selected()

    def browse_arms_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Anatomical File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.arms_file_input.setText(file_path)
            self.check_files_selected()

    def browse_trials_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trials File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.trials_file_input.setText(file_path)
            self.check_files_selected()

    def browse_opensim_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select OpenSim File", "", "OpenSim Files (*.osim);;All Files (*)")
        if file_path:
            self.opensim_file_input.setText(file_path)
            self.check_files_selected()

    def check_files_selected(self):
        if (
            self.trials_file_input.text()
            and self.opensim_file_input.text()
        ):
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)

    def scale_opensim_model(self):
        pass

    def process_data(self):
        if self.processor is None:
            self.processor = ViconProcessor(self.thorax_calibration_file, self.anato_calibration_file)
        self.processor.initialize()
        self.processor.process_trials(self.trials_file_input.text(), self.opensim_file_input.text())

    def closeEvent(self, event):
        event.accept()

    @property
    def thorax_calibration_file(self):
        if self.thorax_file_input.text() == "":
            return None
        return self.thorax_file_input.text()

    @property
    def anato_calibration_file(self):
        if self.arms_file_input.text() == "":
            return None
        return self.arms_file_input.text()

    @property
    def trial_files(self):
        if self.trials_file_input.text() == "":
            return []
        return self.trials_file_input.text().split(";")