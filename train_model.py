import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
from dependencies import fully_connected
from PIL import Image
from glob import glob


#Data directory
train_data_dir = "YOUR_PATH"
test_data_dir = "YOUR_PATH"
pred_data_dir = "YOUR_PATH"

#Fetch the data
for i in os.listdir(train_data_dir):
    new_loc = os.path.join(train_data_dir,i)
    new = new_loc + '/*.jpg'
    images_train = glob(new)
    print(f'{i}:',len(images_train))

for i in os.listdir(test_data_dir):
    new_loc = os.path.join(test_data_dir,i)
    new = new_loc + '/*.jpg'
    images_test = glob(new)
    print(f'{i}:',len(images_test))


for i in range (len(images_train)):
    train_data = []
    img = Image.open(images_train[i], "r")
    train_data.append(np.asarray(img))

for i in range (len(images_test)):
    test_data = []
    img = Image.open(images_test[i], "r")
    test_data.append(np.asarray(img))

#Dictionary for labels
classes = os.listdir(train_data_dir)
classes = {k: v for k,v in enumerate(sorted(classes))}

#Reshape the data
x = train_data.reshape(len(x), 1, 150,150)
x = x.astype("float32")/255
y = test_data.reshape(len(y), 2, 1)

#Feed the data
connected_lay = [
    fully_connected(1, 150, 150),
    dependencies.Softmax(),
    fully_connected(6, 1),
    dependencies.Swish()
]
def guess(connected_lay, a):
    output = a
    for _ in connected_lay:
        output = _.forward(output)
    return output

def train(connected_lay, loss, loss_prime, x, y, iterations = 1000, eta = 0.01, verbose = True):
    for _ in range(iterations):
        err = 0
        for i, j in zip(x, y):
            output = guess(connected_lay, i)
            err += loss(j, output)

            grad = loss_prime(j, output)
            for _ in reversed(connected_lay):
                grad = _.backward(grad, eta)

        err /= len(x)
        if verbose:
            print(f"{_ + 1}/{iterations}, Error={err}")

train(connected_lay, dependencies.av_sqr_err, dependencies.av_sqr_err_prime, x, y, iterations=10000, eta=0.1)

 
res = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = guess(connected_lay, [[x], [y]])
        res.append([x, y, z[0,0]])

res = np.array(res)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(res[:, 0], res[:, 1], res[:, 2], c=res[:, 2])
plt.show()