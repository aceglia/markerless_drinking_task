from multiprocessing import freeze_support
import tkinter as tk
from data_processing.processing_apps.extract_images import App

if __name__ == "__main__":
    freeze_support()

    root = tk.Tk()
    app = App(root)

    root.mainloop()