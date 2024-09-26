# Model training for gunshot classification
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the training and testing dataset
X_train, y_train = load_data('/datasets/train/', '/datasets/labels/train_labels.csv')
X_test, y_test = load_data('/datasets/test/', '/datasets/labels/test_labels.csv')

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('/models/gunshot_classifier.h5')
