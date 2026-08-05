import os
# Automatically detect and add the active Conda environment's library paths
if "CONDA_PREFIX" in os.environ:
    conda_bin = os.path.join(os.environ["CONDA_PREFIX"], "Library", "bin")
    conda_bin_pip = os.path.join(os.environ["CONDA_PREFIX"], "Lib", "site-packages", "nvidia", "cublas", "bin")
    
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)
    if os.path.exists(conda_bin_pip):
        os.add_dll_directory(conda_bin_pip)

import onnxruntime as ort
# Trigger ONNX's built-in DLL preloader if available
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
    
from rtmlib import Wholebody, draw_skeleton

class Wholebody(Wholebody):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def draw_skeleton(self, image, keypoints, scores, kpt_thr=0.5):
        return draw_skeleton(image, keypoints, scores, kpt_thr=kpt_thr)
        

    
