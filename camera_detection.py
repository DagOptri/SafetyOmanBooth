from ultralytics import YOLO
import cv2

# Define the source: 0 for webcam or the path to your video file
VIDEO_SOURCE = 0  # Use "samples/v1.mp4" for a video file
# VIDEO_FILE = "samples/v1.mp4"

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 640  # Adjust based on your needs
resize_height = 480  # Adjust based on your needs
if frame_width > 0:
   resize_height = int((resize_width / frame_width) * frame_height)

skip_frames = 2  # Process every 3rd frame
frame_count = 0

# Load the YOLO model
chosen_model = YOLO("./best.pt")  # Adjust model version as needed

#cap = cv2.VideoCapture(VIDEO_FILE)
def predict(chosen_model, img, classes=[], conf=0.5):
   if classes:
       results = chosen_model.predict(img, classes=classes, conf=conf)
   else:
       results = chosen_model.predict(img, conf=conf)

   return results
def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
   results = predict(chosen_model, img, classes, conf=conf)

   for result in results:
       for box in result.boxes:
           #if lable is person make the box greeen
           if result.names[int(box.cls[0])] == "person":
               cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
               cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
           else:
               cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
               cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
   return img, results

while True:

   success, img = cap.read()

   if not success:
       break
   # Skip frames to speed up processing
   frame_count += 1
   if frame_count % skip_frames != 0:
       continue
   img = cv2.resize(img, (resize_width, resize_height))
   result_img, _ = predict_and_detect(chosen_model, img, classes=[], conf=0.5)

   cv2.imshow("Image", result_img)
  
   cv2.waitKey(1)