import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU

IMGWIDTH = 256

class Meso4:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        model = Model(inputs=x, outputs=y)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, validation_generator, epochs):
        self.model.fit(x_train, validation_data=validation_generator, epochs=epochs, verbose=1)

    def predict_batch(self, x):
        return self.model.predict(x)

    def predict_single(self, img):
        try:
            # Check if the input image is empty or invalid
            if img is None or img.size == 0:
                raise ValueError("Input image is empty or invalid")
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            return self.model.predict(img)
        except Exception as e:
            print("Error:", e)
            return None

    def evaluate_accuracy(self, x_test):
        self.model.evaluate(x_test, verbose=1, max_queue_size=20)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = load_model(filepath)

"""
meso = Meso4()
meso.load_model("C:\Final_Year_Project\Facial_Authentication\Models\Meso4.h5")

img_path = "C:\Final_Year_Project\Sample\Screenshot2024-02-24161208.png"
img = cv2.imread(img_path)

# Predict using the MesoNet model
prediction = meso.predict_single(img)

# Interpret the prediction
if prediction >= 0.80:
    print('Predicted: Real', prediction)
else:
    print('Predicted: Fake', prediction)
"""
