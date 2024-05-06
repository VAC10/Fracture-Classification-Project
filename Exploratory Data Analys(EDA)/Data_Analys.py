# Veri setindeki sınıf sayısını ve her bir sınıftan kaç resmin olduğunu kontrol edelim
fracture_files = os.listdir(fracture_dir)
non_fracture_files = os.listdir(non_fracture_dir)

num_fracture_images = len(fracture_files)
num_non_fracture_images = len(non_fracture_files)

print("Fractured sınıfındaki resim sayısı:", num_fracture_images)
print("Non-fractured sınıfındaki resim sayısı:", num_non_fracture_images)

# Rastgele birkaç resmi görselleştirelim
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(5):
    img_fracture = mpimg.imread(os.path.join(fracture_dir, fracture_files[i]))
    img_non_fracture = mpimg.imread(os.path.join(non_fracture_dir, non_fracture_files[i]))
    
    axes[i].imshow(img_fracture)
    axes[i].set_title("Fractured")
    axes[i].axis("off")
    
    axes[i+5].imshow(img_non_fracture)
    axes[i+5].set_title("Non-fractured")
    axes[i+5].axis("off")

plt.show()

