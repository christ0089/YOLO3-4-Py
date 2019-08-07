import time

from pydarknet import Detector, Image
import cv2

if __name__ == "__main__":

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    cap = cv2.VideoCapture(0)

    while True:
        r, frame = cap.read()
        if r:
            
            
            frame = imutils.resize(frame, width = 1000)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(image)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            print("Elapsed Time:",end_time-start_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            cv2.imshow("preview", frame)


