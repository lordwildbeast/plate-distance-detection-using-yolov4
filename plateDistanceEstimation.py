import cv2 as cv 
import numpy as np
import time

# Distance constants 
KNOWN_DISTANCE = 45 #jarak antara pelat kendaraan dengan kamera dalam cm
PLATE_WIDTH = 27 # lebar pelat, ganti jadi 27 cm

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_SIMPLEX

# getting class names from classes.txt file 
class_names = []
with open("plate.txt", "r") as f: #perlu mengganti isi class.txt menjadi hanya 1 class saja, yaitu plate
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-obj_final.weights', 'yolov4-obj.cfg') #perlu mengganti file .weight dan .cfg

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[] # data_list = [0]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color = COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %.3f" % (class_names[classid], score) #tadinya classid[0]

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # plate class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)]) #tadinya classid[0]
            '''
        elif classid ==67:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)]) #tadinya classid[0]
            '''
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list



# reading the reference image from dir 
ref_plate = cv.imread('ReferencePlateImages/plate6.jpg') #harus upload ref image yang sesuai dengan class untuk hitung width_in_rf (lebar dalam pixel) 

plate_data = object_detector(ref_plate)
plate_width_in_rf = plate_data[0][1] #index out of range terjadi jika ref image tidak dapat dideteksi/tidak sesuai class

print(f"Plate width in pixels : {plate_width_in_rf}")

# finding focal length 
focal_plate = focal_length_finder(KNOWN_DISTANCE, PLATE_WIDTH, plate_width_in_rf)


cap = cv.VideoCapture("video1.mp4")
prev_frame_time = 0
new_frame_time = 0
while True:
    ret, frame = cap.read()
    #gray = frame
    #gray = cv.resize(gray, (600, 400))

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = f"{str(fps)} fps"

    data = object_detector(frame) 
    for d in data:
        if d[0] =='plate':
            distance = distance_finder(focal_plate, PLATE_WIDTH, d[1])
            x, y = d[2]
        
        #cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK, -1) #memberi latar belakang hitam
        cv.putText(frame, f'jarak: {round(distance,2)} cm', (x+150,y-13), FONTS, 0.48, GREEN, 2)

    cv.putText(frame, fps, (10, 50), FONTS, 1, (100, 255, 0), 3, cv.LINE_AA)
    cv.imshow('Estimasi Jarak',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()


