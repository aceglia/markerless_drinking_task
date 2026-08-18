import os

from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QWidget,
    QFileDialog,
)


class ScalingDialog(QDialog):
    """
    Windows for the prefiltering configuration when loading a file. The data rate can be als provided if not included in the file.
    """

    def __init__(self, parent=None, fs=None):
        """
        Initialize the dialog.
        Parameters:
        -----------
        parent: QWidget, optional
            The parent widget of the dialog.
        fs: float, optional
            The sample rate of the data. If not provided, it will be set to the value provided in the file.
        """
        super().__init__(parent)
        self.setWindowTitle("Model Scaling Options")
        self.setGeometry(150, 150, 400, 200)

        self._init_layout()

    def _init_layout(self):
        central_widget = QWidget()
        layout = QGridLayout()
        self.scale_checkbox = QCheckBox("Scale Model")
        layout.addWidget(self.scale_checkbox, 0, 0)
        self.scaling_file_label = QLabel("Trial File:")
        self.scaling_file_input = QLineEdit()
        self.scaling_file_browse_button = QPushButton("Browse")
        self.scaling_file_browse_button.clicked.connect(self.browse_scaling_file)
        layout.addWidget(self.scaling_file_label, 1, 0)
        layout.addWidget(self.scaling_file_input, 1, 1)
        layout.addWidget(self.scaling_file_browse_button, 1, 2)

        self.scaling_options_label = QLabel("Scaling Tool File:")
        self.scaling_options_input = QLineEdit()
        self.scaling_options_browse_button = QPushButton("Browse")
        self.scaling_options_browse_button.clicked.connect(self.browse_scaling_options_file)
        layout.addWidget(self.scaling_options_label, 2, 0)
        layout.addWidget(self.scaling_options_input, 2, 1)
        layout.addWidget(self.scaling_options_browse_button, 2, 2)
        self.ok_button = QPushButton("OK")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button, 3, 0, 1, 4)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button, 3, 0, 1, 4)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

    def _accept(self):
        """
        Accept the dialog and close it.
        """
        self.scaling_file = self.scaling_file_input.text()
        self.scaling_options = self.scaling_options_input.text()
        self.accept()

    def browse_scaling_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trial File", "", "trc Files (*.trc);c3d Files (*.c3d);;All Files (*)")
        if file_path:
            self.scaling_file_input.setText(file_path)
            self.check_files()

    def browse_scaling_options_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scaling Tool File", "", "XML Files (*.xml);;All Files (*)")
        if file_path:
            self.scaling_options_input.setText(file_path)
            self.check_files()

    def check_files(self):
        if self.scaling_file_input.text() and self.scaling_options_input.text():
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)

    def show(self):
        self.scaling_file_input.setText(self.scaling_file)
        self.scaling_options_input.setText(self.scaling_options)
        super().show()