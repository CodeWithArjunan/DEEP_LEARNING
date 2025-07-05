from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]

json_file = open("model.json","r")
load_model_json=json_file.read()
json_file.close()


model=model_from_json(load_model_json)
print('loaded model from disk')

prediction = model.predict(x)
for i in range(10,15):
    print('%s=>%d(expected %d)' %(x[i].tolist(),prediction[i],y[i]))
