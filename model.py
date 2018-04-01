import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

lines = []
with open ('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
lines = lines[1:]
folder = './mydata/IMG/'

for line in lines:
    # center image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = folder + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = folder + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = min(1.0, float(line[3])+0.25)
    measurements.append(measurement)
    
    # right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = folder + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = min(-1.0, float(line[3]) - 0.25)
    measurements.append(measurement)
    
# augment data with fliping images and taking opposite sign of the steering measuremnts
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*(-1.0))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

dropout = 0.5

model = Sequential()

# normalize & cropping
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((65, 25), (0, 0))))

# add 6 convolutional layers
model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))
model.add(Conv2D(24, kernel_size=(5, 5), activation='elu', strides=(2,2)))
model.add(Conv2D(36, kernel_size=(5, 5), activation='elu', strides=(2,2)))
model.add(Conv2D(48, kernel_size=(5, 5), activation='elu', strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

# add a flatten layer
model.add(Flatten())

# add 4 fully connected layers
model.add(Dense(1164, activation='elu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='elu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='elu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='elu'))
model.add(Dropout(dropout))

# add a fully connected output layers
model.add(Dense(1, activation='linear'))

# compile and train the model
optimizer = Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-03, decay=0.0)
model.compile(loss='mse', optimizer=optimizer)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# print model summary
print(model.summary)

# save model
model.save('model.h5')


