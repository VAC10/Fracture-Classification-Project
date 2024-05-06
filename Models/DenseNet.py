from tensorflow.keras.applications.densenet import DenseNet121 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# MobileNet modelini yükleyelim
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) # Ağırlıkların yüklenmesi

# MobileNet modelini dondurarak özellik çıkarma katmanı olarak kullanacağız
mobilenet_model.trainable = False

# Yeni bir model oluşturalım
model_densenet = Sequential()
model_densenet.add(mobilenet_model)
model_densenet.add(Flatten())
model_densenet.add(Dense(16, activation='relu'))
model_densenet.add(Dropout(0.7))
model_densenet.add(Dense(1, activation='sigmoid'))  # Binary sınıflandırma için çıkış katmanı

# Modeli derleyelim
optimizer = Adam(learning_rate=0.00001)  # Öğrenme oranını ayarlayalım
model_densenet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model özetini gösterelim
model_densenet.summary()

# Erken durdurma geri çağrısını tanımlayalım
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)# DenseNet


history = model_densenet.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
