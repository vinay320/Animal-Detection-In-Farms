import cv2
import matplotlib.pyplot as plt
import winsound




thres = 0.5
nms_threshold = 0.5


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classNames = []
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/ 120)
model.setInputMean((120, 120, 120))
model.setInputSwapRB(True)

img = cv2.imread('download.jfif')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidence, bbox = model.detect(img, confThreshold=thres)
print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, confs, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[0]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

cap = cv2.VideoCapture("videoplayback.mp4")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IsADirectoryError("Cannot Open Video")




font_scale = 1
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    # print(ClassIndex)
    for x in ClassIndex:
        print(x);
    if len(ClassIndex)>0:
        dur = 200  # as millisecond
        freq = 1500  # sound frequency
        winsound.Beep(freq, dur)
    if len(ClassIndex)!=0:

        for ClassInd, confs, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd<=80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                text_print = classLabels[ClassInd-1] + " - " + str(round(confs, 2))
                cv2.putText(frame, text_print, (boxes[0]+10, boxes[0]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    cv2.imshow("OUTPUT", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



