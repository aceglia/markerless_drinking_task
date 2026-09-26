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

def process_files(files):
    cpu_count = os.cpu_count()
    num_processes = max(1, cpu_count - 2)

    jobs = [(os.path.dirname(f), os.path.basename(f)) for f in files]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, jobs)

    return results

def process_folder(folder, single_file_path=None):
    cpu_count = os.cpu_count()
    num_processes = max(1, cpu_count - 2)
    files = []
    folders = []
    if folder != "":
        files.extend([f for f in os.listdir(folder) if f.endswith(".db3")])
        folders.extend([f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
    if single_file_path:
        files.append(os.path.basename(single_file_path))
        folders.append(os.path.dirname(single_file_path))

    if not files:
        raise RuntimeError("No .db3 files found in the selected folder.")

    jobs = [(fold, f) for fold, f in zip(folders, files)]
    if len(jobs) == 1:
        results = [process_file(jobs[0])]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_file, jobs)

    return results


class App:

    def __init__(self, root):
        self.root = root

        root.title("RealSense image extractor")
        root.geometry("550x500")
        self.to_process_files = []

        self.folder = tk.StringVar()
        self.single_file_path = tk.StringVar()
        self.match_folder = tk.StringVar()

        # Folder selection
        tk.Label(root, text="Input folder:").pack(pady=(20, 5))
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=20)
        tk.Entry(frame, textvariable=self.folder).pack(side="left", fill="x", expand=True)
        self.fold_button = tk.Button(frame, text="Browse", command=self.select_folder)
        self.fold_button.pack(side="left", padx=(5, 0))
        
        tk.Label(root, text="Input file:").pack(pady=(20, 5))
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=20)
        tk.Entry(frame, textvariable=self.single_file_path).pack(side="left", fill="x", expand=True)
        self.sing_file = tk.Button(frame, text="Browse", command=self.select_single_file)
        self.sing_file.pack(side="left", padx=(5, 0))
        
        tk.Label(root, text="Search for match file in subfolders:").pack(pady=(20, 5))
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=20)
        tk.Entry(frame, textvariable=self.match_folder).pack(side="left", fill="x", expand=True)
        self.find_button = tk.Button(frame, text="Browse", command=self.find_match_files)
        self.find_button.pack(side="left", padx=(5, 0))
        # Process button
        self.process_button = tk.Button(root, text="Process", command=self.process)
        self.process_button.pack(pady=20)

        # Status
        self.status = tk.StringVar(value="Select a folder.")

        tk.Label(root, textvariable=self.status).pack()

    def find_match_files(self):
        self.fold_button.config(state="disabled")
        self.sing_file.config(state="disabled")
        folder = filedialog.askdirectory(title="Select folder where to look for match files")
        if folder:
            self.match_folder.set(folder)
            self.status.set("Match folder selected.")
        self.to_process_files = self.find_match_files_in_subfolders(folder)

    def extract_db3_from_match_files(self, match_files):
        # find what come after : RGBD File Name:
        files = []
        with open(match_files, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "RGBD File Name:" in line:
                    db3_file = line.split("RGBD File Name:")[1].strip()
                    files.append(os.path.join(os.path.dirname(match_files), db3_file))
        return files

    def find_match_files_in_subfolders(self, folder):
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder does not exist.")
            return

        file_to_process = []
        match_files = []
        for root_dir, _, files in os.walk(folder):
            for file in files:
                if file == 'file_match.txt':
                    match_files.append(os.path.join(root_dir, file))
                    file_to_process.extend(self.extract_db3_from_match_files(os.path.join(root_dir, file)))

        if match_files:
            messagebox.showinfo("Match files found", f"Found {len(match_files)} match files.")
            self.status.set(f"Found {len(match_files)} match files, {len(file_to_process)} .db3 files to process.")
        else:
            messagebox.showinfo("No match files found", "No 'file_match.txt' files found in the selected folder.")
            self.status.set("No match files found.")
        return file_to_process

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing .db3 files")

        if folder:
            self.folder.set(folder)
            self.status.set("Folder selected.")

    def select_single_file(self):
        self.find_button.config(state="disabled")
        file_path = filedialog.askopenfilename(title="Select a .db3 file", filetypes=[("DB3 files", "*.db3")])

        if file_path:
            self.single_file_path.set(file_path)
            self.status.set("Single file selected.")

    def process(self):
        self.find_button.config(state="active")
        self.fold_button.config(state="active")
        self.sing_file.config(state="active")

        folder = self.folder.get()
        single_file_path = self.single_file_path.get()

        if not folder and not single_file_path:
            messagebox.showwarning("No folder or file", "Please select a folder or a single file.")
            return

        self.process_button.config(state="disabled")
        self.status.set("Processing...")

        import threading

        thread = threading.Thread(target=self.run_processing, args=(folder, single_file_path, self.to_process_files), daemon=True)
        thread.start()

    def run_processing(self, folder, single_file_path, to_process_files):

        try:
            if len(to_process_files) > 0:
                results = process_files(
                    to_process_files
                )
            else:
                results = process_folder(
                    folder,
                    single_file_path if single_file_path else None

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
