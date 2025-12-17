# ================================
# 1. Importing Required Libraries
# ================================
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.metrics_impl import accuracy


# ================================
# 2. Loading and Preprocessing Data
# ================================
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_images = training_images / 255.0
testing_images = testing_images / 255.0


# ================================
# 3. Defining Class Names
# ================================
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer','Dog','Frog','Horse','Ship','Truck' ]


# ================================
# 4. Visualizing Sample Images
# ================================
for i in range(16):
   plt.subplot(4,4,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.imshow(training_images[i], cmap=plt.cm.binary)
   plt.xlabel(class_names[training_labels[i][0]])

plt.show()


# ================================
# 5. Building the CNN Model
# ================================
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# ================================
# 6. Compiling the Model
# ================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ================================
# 7. Training the Model
# ================================
model.fit(training_images, training_labels,
          epochs=10,
          validation_data=(testing_images, testing_labels))


# ================================
# 8. Evaluating the Model
# ================================
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")


# ================================
# 9. Saving the Model
# ================================
model.save('image_classifier.keras')


# ================================
# 10. Loading the Saved Model
# ================================
model = models.load_model('image_classifier.keras')


# ================================
# 11. Testing an External Image
# ================================
img = cv.imread('')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img, (32, 32))


# ================================
# 12. Making Prediction
# ================================
prediction = model.predict(np.array([img_resized]) / 255.0)
index = np.argmax(prediction)
label = class_names[index]


# ================================
# 13. Displaying the Result
# ================================
plt.figure(figsize=(4, 4))
plt.imshow(img_resized)
plt.title(f'Prediction is : {label}')
plt.axis('off')
plt.show()



