from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

# Load the preprocessed data
X_train = np.load('features/train_features.npy')
y_train = np.load('features/train_labels.npy')
X_test = np.load('features/test_features.npy')
y_test = np.load('features/test_labels.npy')

# Build the neural network model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))  # 40 MFCC features
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (gunshot vs. not)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('models/gunshot_model.h5')

# Evaluate the model and log results
loss, accuracy = model.evaluate(X_test, y_test)
with open('results/model_evaluation.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n")
