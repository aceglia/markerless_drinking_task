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
from .process_utils import ViconProcessor
from .popup_utils import ScalingDialog


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
        self.calibration_files_label = QLabel("Calibration Files:")
        self.calibration_files_input = QLineEdit()
        self.calibration_files_browse_button = QPushButton("Browse")
        self.calibration_files_browse_button.clicked.connect(self.browse_calibration_files)

        file_selection_layout.addWidget(self.calibration_files_label, 0, 0)
        file_selection_layout.addWidget(self.calibration_files_input, 0, 1)
        file_selection_layout.addWidget(self.calibration_files_browse_button, 0, 2)

        self.trials_file_label = QLabel("Trials File:")
        self.trials_file_input = QLineEdit()
        self.trials_file_browse_button = QPushButton("Browse")
        self.trials_file_browse_button.clicked.connect(self.browse_trials_file)

        file_selection_layout.addWidget(self.trials_file_label, 1, 0)
        file_selection_layout.addWidget(self.trials_file_input, 1, 1)
        file_selection_layout.addWidget(self.trials_file_browse_button, 1, 2)

        self.opensim_file_label = QLabel("OpenSim File:")
        self.opensim_file_input = QLineEdit()
        self.opensim_file_browse_button = QPushButton("Browse")
        self.opensim_file_browse_button.clicked.connect(self.browse_opensim_file)

        self.scale_model_button = QPushButton("Model scaling options")
        self.scale_model_button.clicked.connect(self.open_model_scaling_options)

        file_selection_layout.addWidget(self.opensim_file_label, 2, 0)
        file_selection_layout.addWidget(self.opensim_file_input, 2, 1)
        file_selection_layout.addWidget(self.opensim_file_browse_button, 2, 2)
        file_selection_layout.addWidget(self.scale_model_button, 2, 3)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)

        self.option_files_label = QLabel("Options File:")
        self.options_files = QLineEdit()
        self.options_files_browse_button = QPushButton("Browse")
        self.options_files_browse_button.clicked.connect(self.browse_options_file)
        file_selection_layout.addWidget(self.process_button, 3, 0, 1, 4)
        file_selection_layout.addWidget(self.option_files_label, 4, 0)
        file_selection_layout.addWidget(self.options_files, 4, 1)
        file_selection_layout.addWidget(self.options_files_browse_button, 4, 2)

        self.central_widget.setLayout(file_selection_layout)

    def open_model_scaling_options(self):
        if self.scaling_option is None:
            self.scaling_option = ScalingDialog(self)
        if self.scaling_option.exec_() == ScalingDialog.Accepted:
            self.opensim_scale.setChecked(self.scaling_option.scale_checkbox.isChecked())

    def browse_trials_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trials File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.trials_file_input.setText(file_path)
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
            and self.opensim_file_input.text()
        ):
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)
        
    def process_data(self):
        if self.processor is None:
            self.processor = ViconProcessor()
        self.processor.initialize(calibration_files=[self.thorax_calibration_file, self.anato_calibration_file], options_file=self.options_files.text())
        self.processor.batch_process_trials(self.trials_file_input.text(), opensim_model=self.opensim_file_input.text(), scale=self.opensim_scale.isChecked())

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