import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout


def baseline_model():
    model = Sequential()
    model.add(Dense(50, input_dim=400, activation='elu', init='uniform'))
    model.add(Dense(200, input_dim=50, activation='elu', init='uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, input_dim=200, activation='softmax', init='uniform'))

    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return model


def vectorize(train_labels):
    train_labels_vec = []
    for i in train_labels:
        new_vec = np.zeros(10)
        new_vec[i] = 1
        train_labels_vec.append(new_vec)
    return np.array(train_labels_vec)


img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray[gray > 0] = 255

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:,50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = vectorize(np.repeat(k, 250)[:,np.newaxis])
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
model = baseline_model()

model.fit(train, train_labels, nb_epoch=150, verbose=1)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
result = model.predict(test)
matches = np.argmax(result, 1) == np.argmax(test_labels, 1)
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.shape[0]

print matches.shape
print correct
print result.shape
print accuracy

phone = cv2.imread('phone.png')
gray_phone = cv2.cvtColor(phone, cv2.COLOR_BGR2GRAY)
phone_cells = np.hsplit(gray_phone, 7)
phone_cells_np = np.array(phone_cells)
phone_cell_linear = phone_cells_np.reshape(-1, 400).astype(np.float32)

result = model.predict(phone_cell_linear)

print np.argmax(result, 1)
