# Veri Setini Bölme (%70 training %20 testing %10 validation ayrım)
X_train_frac, X_temp_frac, y_train_frac, y_temp_frac = train_test_split(X_fracture, y_fracture, test_size=0.2, random_state=42)
X_val_frac, X_test_frac, y_val_frac, y_test_frac = train_test_split(X_temp_frac, y_temp_frac, test_size=0.1, random_state=42)

X_train_non_frac, X_temp_non_frac, y_train_non_frac, y_temp_non_frac = train_test_split(X_non_fracture, y_non_fracture, test_size=0.2, random_state=42)
X_val_non_frac, X_test_non_frac, y_val_non_frac, y_test_non_frac = train_test_split(X_temp_non_frac, y_temp_non_frac, test_size=0.1, random_state=42)

# Kırık görüntülerin eğitim verisi sayısını eşitledim
X_train_augmented = []
y_train_augmented = []
num_augmented_images_needed = len(X_train_non_frac) - len(X_train_frac)

for X_batch, y_batch in autoaugment.flow(X_train_frac, y_train_frac, batch_size=32): # autoaugment klasörü belirtilmiştir.
    X_train_augmented.append(X_batch)
    y_train_augmented.append(y_batch)
    if len(X_train_augmented) >= num_augmented_images_needed:
        break

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Eşitlenmiş veri kümesini birleştirme işlemi
X_train = np.concatenate((X_train_non_frac, X_train_augmented), axis=0)
y_train = np.concatenate((y_train_non_frac, y_train_augmented), axis=0)
