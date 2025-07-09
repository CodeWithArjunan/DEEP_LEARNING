from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(255,256,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(units=150,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(units=6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=12.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.15,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('HandGestureDataset/train',
                                              target_size=(255,256),
                                              color_mode='grayscale',
                                              batch_size=8,
                                              classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                              class_mode='categorical')

val_set = val_datagen.flow_from_directory('HandGestureDataset/test',
                                              target_size=(255,256),
                                              color_mode='grayscale',
                                              batch_size=8,
                                              classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                              class_mode='categorical')
print("Trained..")

#callback_list =[
#   EarlyStopping(monitor='val_loss',patience=10),
 #   ModelCheckpoin(filepath='model.h5',monitor='val_loss',save_best_only=True,verbose=1)]

callback_list = [
    EarlyStopping(monitor='val_loss',patience=10),
    ModelCheckpoint(filepath="model.h5",monitor='val_loss',save_best_only=True,verbose=1)]

model.fit(train_set,
                    steps_per_epoch=37,
                    epochs=5,
                    validation_data=val_set,
                    validation_steps=7,
                    callbacks=callback_list
                    )




print('save')
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write("model.json")
model.save_weights("model.weights.h5")
print("model saved to disk")

