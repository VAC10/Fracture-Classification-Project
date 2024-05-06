from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Xception modelini yükleyelim
xception_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Xception modelini dondurarak özellik çıkarma katmanı olarak kullanacağız
xception_model.trainable = False

# Yeni bir model oluşturalım
model_xception = Sequential()
model_xception.add(xception_model)
model_xception.add(Flatten())
model_xception.add(Dense(16, activation='relu'))
model_xception.add(Dropout(0.7))
model_xception.add(Dense(1, activation='sigmoid'))  # Binary sınıflandırma için çıkış katmanı


# Modeli derleyelim
optimizer = Adam(learning_rate=0.00001)   # Öğrenme oranını ayarlayalım
model_xception.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model özetini gösterelim
model_xception.summary()

# Erken durdurma geri çağrısını tanımlayalım
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model eğitimini gerçekleştirelim
history = model_xception.fit(
    X_train, y_train,
    epochs=18,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

