import os

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QCheckBox,
)
from .vicon_processor import ViconProcessor
from .popup_utils import ScalingDialog


class ViconProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vicon Processing App")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.scaling_tools_file = None
        self._init_layout()
        self.processor = None

    def _init_layout(self):
        # Create file selection layout
        file_selection_layout = QGridLayout()
        self.calibration_files_label = QLabel("Calibration Files:")
        self.calibration_files_input = QLineEdit()
        self.calibration_files_browse_button = QPushButton("Browse")
        self.calibration_files_browse_button.clicked.connect(self.browse_calibration_files)

        file_selection_layout.addWidget(self.calibration_files_label, 0, 0)
        file_selection_layout.addWidget(self.calibration_files_input, 0, 1)
        file_selection_layout.addWidget(self.calibration_files_browse_button, 0, 2)

        self.trials_file_label = QLabel("Trial Files:")
        self.trials_file_input = QLineEdit()
        self.trials_file_browse_button = QPushButton("Browse")
        self.trials_file_browse_button.clicked.connect(self.browse_trials_file)

        self.rgbd_dir_label = QLabel("RGBD match file: (optional)")
        self.rgbd_file_input = QLineEdit()
        self.rgbd_dir_browse_button = QPushButton("Browse")
        self.rgbd_dir_browse_button.clicked.connect(self.browse_rgbd_dir)

        file_selection_layout.addWidget(self.trials_file_label, 1, 0)
        file_selection_layout.addWidget(self.trials_file_input, 1, 1)
        file_selection_layout.addWidget(self.trials_file_browse_button, 1, 2)

        self.opensim_file_label = QLabel("OpenSim Model (optional):")
        self.opensim_file_input = QLineEdit()
        self.opensim_file_browse_button = QPushButton("Browse")
        self.opensim_file_browse_button.clicked.connect(self.browse_opensim_file)

        # self.scale_model_button = QPushButton("Model scaling options")
        # self.scale_model_button.clicked.connect(self.open_model_scaling_options)

        self.scale_model_file = QPushButton("Scaling setup file (optional)")
        self.scale_model_file.clicked.connect(self.load_scaling_tools)

        file_selection_layout.addWidget(self.opensim_file_label, 2, 0)
        file_selection_layout.addWidget(self.opensim_file_input, 2, 1)
        file_selection_layout.addWidget(self.opensim_file_browse_button, 2, 2)
        # file_selection_layout.addWidget(self.scale_model_button, 2, 3)
        file_selection_layout.addWidget(self.scale_model_file, 2, 3)

        self.option_files_label = QLabel("Options File (optional):")
        self.options_files = QLineEdit()
        self.options_files_browse_button = QPushButton("Browse")
        self.options_files_browse_button.clicked.connect(self.browse_options_file)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)


        file_selection_layout.addWidget(self.option_files_label, 3, 0)
        file_selection_layout.addWidget(self.options_files, 3, 1)
        file_selection_layout.addWidget(self.options_files_browse_button, 3, 2)
        file_selection_layout.addWidget(self.rgbd_dir_label, 4, 0)
        file_selection_layout.addWidget(self.rgbd_file_input, 4, 1)
        file_selection_layout.addWidget(self.rgbd_dir_browse_button, 4, 2)

        file_selection_layout.addWidget(self.process_button, 5, 0, 1, 4)


        self.central_widget.setLayout(file_selection_layout)

    def load_scaling_tools(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scaling Tools File", "", "XML Files (*.xml);;All Files (*)")
        if file_path:
            self.scaling_tools_file = file_path
            print(f"Loaded scaling tools from: {file_path}")

    def open_model_scaling_options(self):
        if self.scaling_option is None:
            self.scaling_option = ScalingDialog(self)
        if self.scaling_option.exec_() == ScalingDialog.Accepted:
            self.opensim_scale.setChecked(self.scaling_option.scale_checkbox.isChecked())

    def extract_db3_from_match_files(self, match_files):
        vicon_file = None
        db3_file = None
        files = {}
        with open(match_files, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Vicon File Name:" in line:
                    vicon_file = line.split("Vicon File Name:")[1].split('.x2d')[0].strip()
                if "RGBD File Name:" in line:
                    db3_file = line.split("RGBD File Name:")[1].split('.db3')[0].strip()
                if vicon_file and db3_file:
                    files[vicon_file] = os.path.join(os.path.dirname(match_files), db3_file)
        return files

    def browse_rgbd_dir(self):
        match_file_path, _ = QFileDialog.getOpenFileName(self, "Select RGBD match file", "", "Text Files (*.txt);;All Files (*)")
        if match_file_path:
            self.rgbd_file_input.setText(match_file_path)
            self.check_files_selected()

    def browse_calibration_files(self):
        file_path, _ = QFileDialog.getOpenFileNames(self, "Select Calibration Files", "", "C3D Files (*.c3d);;All Files (*)")
        if file_path:
            self.calibration_files_input.setText(";".join(file_path))
            self.check_files_selected()

    def browse_trials_file(self):
        file_path, _ = QFileDialog.getOpenFileNames(self, "Select Trials File", "", "C3D Files (*.c3d);;All Files (*)")
        if file_path:
            self.trials_file_input.setText(";".join(file_path))
            self.check_files_selected()

    def browse_opensim_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select OpenSim Model File", "", "OpenSim Files (*.osim);;All Files (*)")
        if file_path:
            self.opensim_file_input.setText(file_path)
            self.check_files_selected()

    def browse_options_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Options File", "", "YAML Files (*.yaml);;All Files (*)")
        if file_path:
            self.options_files.setText(file_path)
            self.check_files_selected()

    def check_files_selected(self):
        if (
            self.trials_file_input.text()
            and self.calibration_files_input.text()
        ):
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)
        
    def _threaded_process_data(self):
        if self.processor is None:
            self.processor = ViconProcessor()
        self.processor.initialize(calibration_files=self.calibration_files, options_file=self.options_files.text())
        osim_model = r"opensim/wu_modified_vicon.osim" if self.opensim_file_input.text() == "" else self.opensim_file_input.text()
        scaling_tool = r"opensim/scaling_tool_vicon.xml" if self.scaling_tools_file is None else self.scaling_tools_file
        trials_file = [self.trials_file_input.text()] if isinstance(self.trials_file_input.text(), str) else self.trials_file_input.text()
        matching_files = self.extract_db3_from_match_files(self.rgbd_file_input.text()) if self.rgbd_file_input.text() != "" else []
        self.processor.batch_process_trials(trials_file, opensim_model=osim_model, scale=scaling_tool, rgbd_matching_files=matching_files)
        self.process_button.setEnabled(True)

    def process_data(self):
        import threading
        self.process_button.setEnabled(False)
        self.process_thread = threading.Thread(target=self._threaded_process_data, daemon=True)
        self.process_thread.start()

    def closeEvent(self, event):
        event.accept()

    @property
    def calibration_files(self):
        if self.calibration_files_input.text() == "":
            return []
        return self.calibration_files_input.text().split(";")

    @property
    def trial_files(self):
        if self.trials_file_input.text() == "":
            return []
        return self.trials_file_input.text().split(";")