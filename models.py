from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tcn import TCN

# Define the size of the CQT slices
cqt_slice_size = 5  # seconds

# Define the size of the chord label windows
chord_label_window_size = 30  # seconds

# Define the overlap between chord label windows
chord_label_window_overlap = 15  # seconds

# Slice the audio into CQT windows
cqt_windows = window_audio(audio, sr, cqt_slice_size, cqt_slice_size/2)

# Create the corresponding chord label windows
chord_labels_windows = window_chord_labels(chord_labels, sr, chord_label_window_size, chord_label_window_overlap)

# Create a dataset of CQT slices and chord labels
dataset = list(zip(cqt_windows, chord_labels_windows))

# Convert the chord labels to integers with range -1 to 23
dataset = [(cqt_slice, chord_to_number(chord_labels_window)) for cqt_slice, chord_labels_window in dataset]

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the CNN model
cnn_model = Sequential([
    Input(shape=(cqt_slice_size*sr, 1)),
    Reshape((cqt_slice_size*sr, 1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
])

# Define the TCN model
tcn_model = Sequential([
    Input(shape=(cqt_slice_size*sr, 1)),
    TCN(nb_filters=64, kernel_size=3, dropout_rate=0.2, activation='relu'),
    TCN(nb_filters=128, kernel_size=3, dropout_rate=0.2, activation='relu'),
    Dense(25, activation='softmax')
])

# Compile both models
cnn_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(),
                  metrics=[SparseCategoricalAccuracy()])
tcn_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(),
                  metrics=[SparseCategoricalAccuracy()])

# Train both models
history_cnn = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_val_cnn, y_val_cnn))
history_tcn = tcn_model.fit(X_train_tcn, y_train_tcn, epochs=10, validation_data=(X_val_tcn, y_val_tcn))

# Evaluate both models on the test set
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_cnn)
tcn_test_loss, tcn_test_acc = tcn_model.evaluate(X_test_tcn, y_test_tcn)

# Make predictions on the test set with both models
cnn_predictions = cnn_model.predict(X_test_cnn)
tcn_predictions = tcn_model.predict(X_test_tcn)

# Print the test set accuracies and prediction shapes
print("CNN Test Accuracy:", cnn_test_acc)
print("TCN Test Accuracy:", tcn_test_acc)
print("CNN Predictions Shape:", cnn_predictions.shape)
print("TCN Predictions Shape:", tcn_predictions.shape)