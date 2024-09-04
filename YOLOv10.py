from ultralytics  import YOLO

model  = YOLO('YOLOv10_model/best.pt')
source = 'test1.mp4'
model.predict(source = source, save = True)                                                                                                                                 