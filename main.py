
import os
import cv2
from ultralytics import YOLO
import torch

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.device(0)



def face_detection():
    model = YOLO(r"E:\Università Magistrale\Tesi\Materiali Tesi\Dataset_celeba\yolov8n-face.pt")

    parent_folder = f'D:/img_align_celeba'

    output_folder = f'D:/img_align_celeba_crop'

    os.makedirs(output_folder, exist_ok=True)

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        if os.path.isdir(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    img_full_path = os.path.join(root, filename)
                    results = model.predict(img_full_path)

                    img = cv2.imread(img_full_path)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cords = box.xyxy[0].tolist()
                            cords = [round(x) for x in cords]
                            top_x, top_y, bot_x, bot_y = cords
                            cv2.rectangle(img, (top_x, top_y), (bot_x, bot_y), (57, 255, 20), 1)
                            face_roi = img[top_y + 1:bot_y - 1, top_x + 1:bot_x - 1]
                            file_name = os.path.splitext(os.path.basename(img_full_path))[0]
                            cv2.imwrite(os.path.join(output_folder, f"{file_name}.png"), face_roi)
                          
face_detection()
