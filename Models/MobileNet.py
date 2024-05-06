from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# MobileNet modelini yükleyelim
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# MobileNet modelini dondurarak özellik çıkarma katmanı olarak kullanacağız
mobilenet_model.trainable = False

# Yeni bir model oluşturalım
model_mobilenet = Sequential()
model_mobilenet.add(mobilenet_model)
model_mobilenet.add(Flatten())
model_mobilenet.add(Dense(16, activation='relu'))
model_mobilenet.add(Dropout(0.7))
model_mobilenet.add(Dense(1, activation='sigmoid'))  # Binary sınıflandırma için çıkış katmanı

# Modeli derleyelim
optimizer = Adam(learning_rate=0.00001)  # Öğrenme oranını ayarlayalım
model_mobilenet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model özetini gösterelim
model_mobilenet.summary()

# Erken durdurma geri çağrısını tanımlayalım
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
