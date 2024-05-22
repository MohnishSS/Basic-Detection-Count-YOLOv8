from ultralytics import YOLO
import cv2

#change image source here
#change show=False to True to see the image with objects highlighted
model = YOLO('yolov8l.pt')
results = model("sample-images/cars.png", show=False)

objects=[]

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

for r in results:
    boxes=r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        if(box.conf[0]>=0.5):
                objects.append(classNames[cls])

objects_in_image=list(set(objects))
for obj in objects_in_image:
    count=0
    for i in objects:
        if i==obj:
                count+=1
    print(str(count) +" "+ obj + ("s" if count>1 else ""))
