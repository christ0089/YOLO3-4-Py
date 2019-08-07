import time

from pydarknet import Detector, Image
import cv2
import imutils




def postProcess(img):
    config = {"y_offset": 20, # maximum y offset between chars
        "x_offset":  55, # maximum x gap between chars
            "thesh_offset":  0, # this determines the cutoff point on the adaptive threshold.
            "thesh_window": 25, # window of adaptive theshold area
            # max min char width, height and ratio
            "w_min":  6, # char pixel width min
            "w_max":  30, # char pixel width max
            "h_min":  12, # char pixel height min
            "h_max":  40, # char pixel height max
            "hw_min":  1.5, # height to width ration min
            "hw_max":  3.5, # height to width ration max
            "h_ave_diff":  1.09,  # acceptable limit for variation between characters
    }
    #cv2.imshow("Input",img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('path', metavar='video_path', type=str,
                        help='Path to source video')

    args = parser.parse_args()
    print("Source Path:", args.path)
    cap = cv2.VideoCapture(args.path)


    average_time = 0

    net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))
    count = 0

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()
            # Only measure the time taken by YOLO and API Call overhead
            frame = imutils.resize(frame, width=1000)
            dark_frame = Image(frame)
            
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time-start_time) * 0.2
            
            print("Total Time:", end_time-start_time, ":", average_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            cv2.imshow("preview", frame)
    #print ('Number of Frames:', cap.get(cv2.CAP_PROP_FRAME_COUNT))


        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
