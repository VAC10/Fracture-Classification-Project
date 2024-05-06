from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Resimleri ve etiketlerini yüklemek için kullanılacak listelerin tanımlanması
images = []
labels = []

# Kırık resimleri yükleme ve etiketleme
for filename in os.listdir(fracture_dir):
    img = load_img(os.path.join(fracture_dir, filename), target_size=(224, 224))  # Resmi yükleme ve boyutlandırma
    img_array = img_to_array(img) / 255.0  # Resmi numpy dizisine dönüştürme ve normalize etme işlemi
    images.append(img_array)  # Resmi listeye ekleme
    labels.append(1)  # Kırık olduğu belirtilen etiketi ekleme

# Kırık olmayan resimleri yükleme ve etiketleme
for filename in os.listdir(non_fracture_dir):
    img = load_img(os.path.join(non_fracture_dir, filename), target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  
    images.append(img_array)  # Resmi listeye ekleme
    labels.append(0)  # Kırık olmadığını belirten etiketi ekleme

# Resim ve etiket listelerini numpy dizilerine (tensör) dönüştürme
X = np.array(images)
y = np.array(labels)
