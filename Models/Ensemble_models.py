from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# --DenseNet--- modelini yükleyelim ve donduralım
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
densenet_base.trainable = False

# Özellik çıkarma modeli oluşturma ve özellikleri düzleştirme
densenet_feature_extractor = Sequential([densenet_base, GlobalAveragePooling2D()])
densenet_features_train = densenet_feature_extractor.predict(X_train)
densenet_features_test = densenet_feature_extractor.predict(X_test)



from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# --MobileNet--- modelini yükleyelim ve donduralım
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_base.trainable = False

# Özellik çıkarma modeli oluşturma ve özellikleri düzleştirme
mobilenet_feature_extractor = Sequential([mobilenet_base, GlobalAveragePooling2D()])
mobilenet_features_train = mobilenet_feature_extractor.predict(X_train)
mobilenet_features_test = mobilenet_feature_extractor.predict(X_test)

# ViT
import tensorflow as tf
from tensorflow.keras import layers

# Patches (Yamalar) Sınıfı
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images, patch_size):  # patch_size argümanı eklendi
        batch_size = tf.shape(images)[0] # Giriş görüntülerinin batch boyutunu alır.
        patches = tf.image.extract_patches( # görüntülerden yamalar çıkarılır.
            images=images,
            sizes=[1, patch_size, patch_size, 1],  # patch_size kullanıldı
            strides=[1, patch_size, patch_size, 1],  # patch_size kullanıldı
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1] #  Çıkarılan yamaların boyutunu alır.
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #yamaların boyutunu düzenler ve yeniden şekillendirir
        return patches

# Patch Encoder (Yama Kodlayıcı) Sınıfı
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches = self.projection(patches) + self.position_embedding(positions)
        return encoded_patches

# Basit Vision Transformer Modeli Oluşturma
def create_simple_vit_classifier(image_size, patch_size, num_classes, num_patches, projection_dim, transformer_layers, num_heads, transformer_units, mlp_head_units, dropout_rate):
    inputs = layers.Input(shape=image_size + (3,))
    
    # Giriş görüntülerinden yamaları çıkarın
    patches = Patches(patch_size)(inputs, patch_size=patch_size) 
    
    # Patch Encoder
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer Bloğu
    for _ in range(transformer_layers):
        # Dikkat Katmanı
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=transformer_units, activation="gelu")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(units=projection_dim)(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Temsil (Representation) Katmanı
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    
    # MLP Başlık Katmanı
    features = layers.Dense(units=mlp_head_units, activation="gelu")(representation)
    features = layers.Dropout(dropout_rate)(features)
    
    # Çıkış Katmanı
    logits = layers.Dense(units=num_classes, activation="sigmoid")(features)
    
    # Model Oluşturma
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# Modeli Oluştur
image_size = (224, 224)
patch_size = 16
num_classes = 1  
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
projection_dim = 32
transformer_layers = 2  
num_heads = 2
transformer_units = 1024  
mlp_head_units = 512
dropout_rate = 0.7  

# Modeli oluştur
simple_vit_classifier = create_simple_vit_classifier(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    num_patches=num_patches,
    projection_dim=projection_dim,
    transformer_layers=transformer_layers,
    num_heads=num_heads,
    transformer_units=transformer_units,
    mlp_head_units=mlp_head_units,
    dropout_rate=dropout_rate
)

# Modeli derleme
simple_vit_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.BinaryCrossentropy(),  
    metrics=[tf.keras.metrics.BinaryAccuracy()]  
)

# Özellikleri çıkar
vit_features_train = simple_vit_classifier.predict(X_train)
vit_features_test = simple_vit_classifier.predict(X_test)


#3 MODELİN SONUÇLARINI BİRLEŞTİRME
import numpy as np

# Eğitim verisi için özellikleri birleştirme
combined_features_train = np.concatenate(
    (densenet_features_train, mobilenet_features_train,vit_features_train), axis=1
)

# Test verisi için özellikleri birleştirme
combined_features_test = np.concatenate(
    (densenet_features_test, mobilenet_features_test, vit_features_test), axis=1
)

# 3 MODELİN GİRİŞİNİ BİR NEURAL NETWORK İLE TEKRAR EĞİTME İŞLEMİ
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Yeni bir model oluşturma
ensemble_model = Sequential(name="ensemble_model")
ensemble_model.add(Dense(1024, activation='relu', input_shape=(combined_features_train.shape[1],)))
ensemble_model.add(Dropout(0.6))
ensemble_model.add(Dense(512, activation='relu', input_shape=(combined_features_train.shape[1],)))
ensemble_model.add(Dropout(0.6))

ensemble_model.add(Dense(1, activation='sigmoid'))  # Binary sınıflandırma için

# Modeli derleme
ensemble_model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Model özetini gösterme
ensemble_model.summary()

# Erken durdurma geri çağrısını tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitme
history_ensemble = ensemble_model.fit(combined_features_train, y_train, epochs=40, batch_size=32, validation_data=(combined_features_test, y_test), callbacks=[early_stopping])










