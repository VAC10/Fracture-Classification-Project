from sklearn.model_selection import train_test_split
import numpy as np
import os


# Resimleri ve etiketlerini yüklemek için kullanılacak listelerin tanımlanması
images_fracture = []
labels_fracture = []
images_non_fracture = []
labels_non_fracture = []

# Kırık resimleri yükleme ve etiketleme
for filename in os.listdir(fracture_dir):
    img = load_img(os.path.join(fracture_dir, filename), target_size=(224, 224))  # Resmi yükleme ve boyutlandırma
    img_array = img_to_array(img) / 255.0  # Resmi numpy dizisine dönüştürme ve normalize etme işlemi
    images_fracture.append(img_array)  # Resmi listeye ekleme
    labels_fracture.append(1)  # Kırık olduğu belirtilen etiketi ekle

# Kırık olmayan resimleri yükleme ve etiketleme
for filename in os.listdir(non_fracture_dir):
    img = load_img(os.path.join(non_fracture_dir, filename), target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  
    images_non_fracture.append(img_array)  # Resmi listeye ekle
    labels_non_fracture.append(0)  # Kırık olmadığını belirten etiketi ekle

# Resim ve etiket listelerini numpy dizilerine (tensör) dönüştürme
X_fracture = np.array(images_fracture)
y_fracture = np.array(labels_fracture)
X_non_fracture = np.array(images_non_fracture)
y_non_fracture = np.array(labels_non_fracture)

