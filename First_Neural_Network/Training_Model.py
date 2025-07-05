#importt packages
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

#stored Datase
dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')
print(dataset)

x=dataset[:,0:8] #input
y=dataset[:,8]   #output

model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=10,batch_size=10)

#Evalution
_,accuracy = model.evaluate(x,y)
print('Accurancy : %.2f' %(accuracy*100))


#Save Model
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to dis")
