# main.py

# 📦 Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Add, Dropout
from tensorflow.keras.optimizers import Adam

# 📍 Check current directory
print("Current working directory:", os.getcwd())

# 📥 Load dataset
data_path = "data/genetic_sample.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please make sure the file exists.")

data = pd.read_csv(data_path)
print("Data shape:", data.shape)
print(data.head())

# 🔍 Preprocessing
X = data.drop('label', axis=1).values
y = data['label'].values

# Normalize gene expressions
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 📐 Reshape for Conv1D: (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 🧠 Build ResNet + BiLSTM Model
input_layer = Input(shape=(X_train.shape[1], 1))

# ResNet block
conv1 = Conv1D(64, 3, padding='same', activation='relu')(input_layer)
conv2 = Conv1D(64, 3, padding='same', activation='relu')(conv1)
res_block = Add()([conv1, conv2])  # Residual connection

# BiLSTM
bilstm = Bidirectional(LSTM(64))(res_block)
dropout = Dropout(0.5)(bilstm)
output_layer = Dense(len(np.unique(y_encoded)), activation='softmax')(dropout)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🏋️ Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# 🧪 Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))

# 📊 Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

