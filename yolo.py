import torch

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
ROBOFLOW_API_KEY = api.get_secret("ROBOFLOW_API")

from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("zzigmug").project("fruits-and-vegetables-knetf")
dataset = project.version(3).download("yolov8")