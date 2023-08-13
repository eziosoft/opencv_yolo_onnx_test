from secret import *

from roboflow import Roboflow
rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("dronevision").project("helipad")
dataset = project.version(1).download("yolov8")
