#

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

class_names = [
    'Plane', 'Car', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

model = models.load_model('image_classifier.keras')

img = cv.imread('bird22.jpg')
if img is None:
    raise FileNotFoundError("الصورة غير موجودة، تأكدي من الاسم أو المسار")

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_resized = cv.resize(img, (32, 32))

prediction = model.predict(np.array([img_resized]) / 255.0)
index = np.argmax(prediction)
label = class_names[index]

plt.figure(figsize=(4, 4))
plt.imshow(img_resized)
plt.title(f'Prediction is : {label}')
plt.axis('off')

plt.show()
