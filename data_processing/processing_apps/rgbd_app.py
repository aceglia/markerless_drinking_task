from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QCheckBox,
    QGridLayout,
)

from .rgbd_processor import RGBDProcessor

class RGBDProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGBD Processing App")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._init_layout()
        self.processor = RGBDProcessor(bag_file_path=None)

    def _init_layout(self):
        file_selection_layout = QGridLayout()

        self.trials_file_label = QLabel("Trials File:")
        self.trials_file_input = QTextEdit()
        self.trials_file_input.setReadOnly(True)
        self.trials_file_input.setMaximumHeight(100)
        self.trials_file_browse_button = QPushButton("Browse")
        self.trials_file_browse_button.clicked.connect(self.browse_trials_file)

        file_selection_layout.addWidget(self.trials_file_label, 1, 0)
        file_selection_layout.addWidget(self.trials_file_input, 1, 1)
        file_selection_layout.addWidget(self.trials_file_browse_button, 1, 2)
        self.pose_estimation_box = QCheckBox("Enable Pose Estimation")
        self.pose_estimation_box.setChecked(True)
        file_selection_layout.addWidget(self.pose_estimation_box, 2, 0, 1, 2)
        self.kinematics_box = QCheckBox("Enable Kinematics Computation")
        self.kinematics_box.setChecked(True)
        file_selection_layout.addWidget(self.kinematics_box, 2, 2, 1, 2)
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)

        self.option_files_label = QLabel("Options File (optional):")
        self.options_file = QLineEdit()
        self.options_file_browse_button = QPushButton("Browse")
        self.options_file_browse_button.clicked.connect(self.browse_options_file)
        file_selection_layout.addWidget(self.process_button, 3, 0, 1, 4)
        file_selection_layout.addWidget(self.option_files_label, 4, 0)
        file_selection_layout.addWidget(self.options_file, 4, 1)
        file_selection_layout.addWidget(self.options_file_browse_button, 4, 2)

        self.central_widget.setLayout(file_selection_layout)

    def browse_trials_file(self):
        file_path, _ = QFileDialog.getOpenFileNames(self, "Select Trials File", "", "RealSense Files (*.db3);;RealSense Files (*.bag);;All Files (*)")
        if file_path:
            self.trials_file_input.clear()
            self.trials_file_input.setText("\n".join(file_path))
            self.check_files_selected()

    def browse_options_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Options File", "", "YAML Files (*.yaml);;All Files (*)")
        if file_path:
            self.options_file.setText(file_path)
            self.processor.options.from_file(file_path)
            self.check_files_selected()

    def cancel_processing(self):
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1)
            if self.process_thread.is_alive():
                print("Processing thread is still running. Please wait for it to finish.")
            else:
                print("Processing thread has been terminated.")

    def check_files_selected(self):
        if (
            self.trials_file_input.toPlainText() != ""
        ):
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)

    def _threaded_process_data(self):
        # try:
        self.processor.process_trials(
                path_list=self.trial_files,
                options_file_path=self.options_file.text(),
                run_pose_estimation=self.pose_estimation_box.isChecked(),
                run_kinematics=self.kinematics_box.isChecked()
            )
        self.process_button.setEnabled(True)
        # except Exception as e:
        #     print(f"Error occurred while processing data: {e}")
        
    def process_data(self):
        import threading
        self.process_button.setEnabled(False)
        self.process_thread = threading.Thread(target=self._threaded_process_data, daemon=True)
        self.process_thread.start()

    def closeEvent(self, event):
        event.accept()

    @property
    def trial_files(self):
        if self.trials_file_input.toPlainText() == "":
            return []
        return self.trials_file_input.toPlainText().split("\n")