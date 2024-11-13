#!pip install ultralytics --upgrade

from ultralytics import YOLO

#model = YOLO('yolo11x.pt', task="object") # task defualt = "object"
#model = YOLO('yolo11x-seg.pt', task="segment")
#model =  YOLO('yolo11x-cls.pt', task="classify")
model =  YOLO('yolo11x-pose.pt', task="pose")

results = model("assets/1.jpg", show=True, save=True, save_dir="./output")



