import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load CSV
df = pd.read_csv('fer2013.csv')
pixels = df['pixels'].tolist()
X = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype='float32')
X = X.reshape(-1, 48, 48, 1) / 255.0

y = to_categorical(df['emotion'], num_classes=7)

# Train-val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint('expression_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train
model.fit(X_train, y_train, epochs=20, batch_size=64,
          validation_data=(X_val, y_val), callbacks=[checkpoint])
