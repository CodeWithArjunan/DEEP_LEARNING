from keras.models import Sequential
from keras.layers import Input,Conv2D
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import json


print('load json')
json_file = open('model.json','r')
load=json_file.read()
json_file.close()
model=model_from_json(load,custom_objects={"Sequential":Sequential})
model.load_weights('model.h5')
print('loaded to dusk')

def classify(img_file):
    img_name=img_file
#    test_img=image.load_img(img_name,target_size=(64,64))
    test_img = image.load_img(img_name, target_size = (64, 64))

    test_image = image.img_to_array(test_img)
    test_img=np.expand_dims(test_img,axis=0)
    result = model.predict(test_img)
    
    if result[0][0] == 1:
        predict='Thanos'
    else:
        predict='Joker'
    print(predict,img_name)

   
import os

path='Dataset/test'
print("here are the file I see:",os.listdir("."))
files=[]

#r=root,d=derectory,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' in file:
            files.append(os.path.join(r, file))
    
           
for f in file:
    classify(f)
    print('\n')
