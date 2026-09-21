import os
import tkinter as tk
from tkinter import filedialog, messagebox
from multiprocessing import Pool
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

from data_processing.camera_converter import CameraConverter


def process_file(args):
    dir_path, file = args

    print(f"Processing: {file}")
    file_name = Path(file).stem
    bag_path = os.path.join(dir_path, file)

    dir_path = os.path.join(dir_path, file_name)
    tmp_path = os.path.join(dir_path, "images")
    
    os.makedirs(tmp_path, exist_ok=True)
    pipeline = rs.pipeline()

    try:
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

        profile = pipeline.start(config)

        converter = CameraConverter(use_camera=True)
        converter.set_intrinsics(pipeline)
        converter.set_extrinsics(pipeline)

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        align = rs.align(rs.stream.color)

        count = 0

        # create a json file to store start and stop
        dict = {"range": [-1, -1]}
        with open(os.path.join(dir_path, "range.json"), "w") as json_file:
            json.dump(dict, json_file, indent=4)

        while True:
            try:
                frames = pipeline.wait_for_frames(1500)
            except RuntimeError:
                break

            frames = align.process(frames)

            frame_number = frames.frame_number

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # depth_raw = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            color_path = os.path.join(tmp_path, f"color_{frame_number}.png")

            # depth_path = os.path.join(
            #     tmp_path,
            #     f"depth_{frame_number}.png"
            # )

            if not os.path.exists(color_path):
                cv2.imwrite(color_path, color_image)

            # if not os.path.exists(depth_path):
            #     cv2.imwrite(depth_path, depth_raw)

            # Collect accelerometer data only for first 50 frames
            if count < 50:
                converter.add_accel_frame(frames)

            count += 1

        converter.save_config(os.path.join(dir_path, "camera_config.json"))

        return file, count, None

    except Exception as e:
        return file, 0, str(e)

    finally:
        pipeline.stop()


def process_folder(folder):
    cpu_count = os.cpu_count()
    num_processes = max(1, cpu_count - 2)
    files = [f for f in os.listdir(folder) if f.endswith(".db3")]

    if not files:
        raise RuntimeError("No .db3 files found in the selected folder.")

    jobs = [(folder, f) for f in files]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, jobs)

    return results


class App:

    def __init__(self, root):
        self.root = root

        root.title("RealSense image extractor")
        root.geometry("550x220")

        self.folder = tk.StringVar()

        # Folder selection
        tk.Label(root, text="Input folder:").pack(pady=(20, 5))

        frame = tk.Frame(root)
        frame.pack(fill="x", padx=20)

        tk.Entry(frame, textvariable=self.folder).pack(side="left", fill="x", expand=True)

        tk.Button(frame, text="Browse", command=self.select_folder).pack(side="left", padx=(5, 0))

        # Process button
        self.process_button = tk.Button(root, text="Process", command=self.process)
        self.process_button.pack(pady=20)

        # Status
        self.status = tk.StringVar(value="Select a folder.")

        tk.Label(root, textvariable=self.status).pack()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing .db3 files")

        if folder:
            self.folder.set(folder)
            self.status.set("Folder selected.")

    def process(self):

        folder = self.folder.get()

        if not folder:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return

        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder does not exist.")
            return

        self.process_button.config(state="disabled")
        self.status.set("Processing...")

        import threading

        thread = threading.Thread(target=self.run_processing, args=(folder,), daemon=True)
        thread.start()

    def run_processing(self, folder):

        try:
            results = process_folder(
                folder,
            )

            successful = 0
            failed = 0

            for file, count, error in results:
                if error:
                    print(f"ERROR: {file}: {error}")
                    failed += 1
                else:
                    print(f"Finished: {file} ({count} frames)")
                    successful += 1

            self.root.after(0, lambda: self.processing_finished(successful, failed))

        except Exception as e:

            self.root.after(0, lambda: self.processing_error(str(e)))

    def processing_finished(self, successful, failed):

        self.process_button.config(state="normal")

        self.status.set(f"Finished: {successful} successful, {failed} failed.")

        messagebox.showinfo("Processing finished", f"Successful: {successful}\n" f"Failed: {failed}")

    def processing_error(self, error):

        self.process_button.config(state="normal")

        self.status.set("Error.")

        messagebox.showerror("Processing error", error)
