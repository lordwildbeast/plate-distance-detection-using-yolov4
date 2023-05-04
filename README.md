# Description
Distance estimation of Indonesian license plate using Yolov4

# To-do
## Installation you need opencv-contrib-python

[opencv contrib](https://pypi.org/project/opencv-contrib-python/)

### **Windows**

```pip
pip install opencv-contrib-python==4.5.3.56
```

### **Linux or Mac**

```pip
pip3 install opencv-contrib-python==4.5.3.56
```

---

## Add more Classes(Objects) for Distance Estimation

You will make changes on these particular lines [***plateDistanceEstimation.py***](https://github.com/lordwildbeast/plate-distance-detection-using-yolov4/blob/main/plateDistanceEstimation.py#L59-L67)
```python
if classid ==0: # person class id 
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
elif classid ==67: # cell phone
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
    
# adding more classes for distnaces estimation 

elif classid ==2: # car
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])

elif classid ==15: # cat
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
# in that way you can include as many classes you want 

    # returning list containing the object data. 
return data_list

```

## Reading images and getting focal length

You have to make changes on these lines 📝 [***plateDistanceEstimation.py***](https://github.com/lordwildbeast/plate-distance-detection-using-yolov4/blob/main/plateDistanceEstimation.py#L72-L75)

```python
# reading refrence images 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')
# calling the object detector function to get the width or height of object
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

# getting pixel width for cat
cat_data = object_detector(ref_person)
cat_width_in_rf = person_data[2][1]

# getting pixel width for car
car_data = object_detector(ref_person)
car_width_in_rf = person_data[3][1]

```
if there is single class(object) in reference image then you approach it that way 👍
```python
# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/person_ref_img.png')
ref_car = cv.imread('ReferenceImages/car_ref_img.png.png')
ref_cat = cv.imread('ReferenceImages/cat_ref_img.png')
ref_mobile = cv.imread('ReferenceImages/ref_cell_phone.png')

# checking object detection on reference image 
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

# getting pixel width for cat
cat_data = object_detector(ref_cat)
cat_width_in_rf = person_data[0][1]

# getting pixel width for car
car_data = object_detector(ref_car)
car_width_in_rf = person_data[0][1]
# then you find Focal length for each

```

# References
YOLOv4 in the CLOUD: Build and Train Custom Object Detector (FREE GPU) [Youtube](https://youtu.be/mmj3nxGT2YQ)

Distance Estimation using |Single camera | YoloV4 Object Detector 
[Youtube](https://youtu.be/FcRCwTgYXJw)
[Github](https://github.com/Asadullah-Dal17/Yolov4-Detector-and-Distance-Estimator)
