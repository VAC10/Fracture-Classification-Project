# X-Ray görüntülerinden kırık kemik sınıflandırması
Bu proje, derin öğrenme yöntemlerini kullanarak radyolojik görüntülerdeki kemik kırıklarını sınıflandırmak için geliştirilmiştir. Kemik kırıklarının doğru bir şekilde sınıflandırılması, tıbbi teşhis süreçlerinde önemli bir rol oynayabilir, Hızlı ve güvenilir tanı için destek sağlar ve tedavi planlaması için değerli bilgiler verebilir.



# Amaç
Bu projenin ana amacı, derin öğrenme ve görüntü işleme tekniklerini kullanarak radyolojik görüntülerdeki kemik kırıklarını otomatik olarak sınıflandırmaktır. Bu sınıflandırma işlemi, farklı kemik kırığı tiplerinin tanımlanmasını ve teşhis edilmesini kolaylaştırarak sağlık profesyonellerine değerli bir araç sunabilir.

# Kullanılan Teknolojiler
Bu model, Kaggle'a Nvidia tarafından sunulan, yüksek performanslı GPU "P100" ile eğitilmiştir.

# Veri Kümesi
Bu projede kullanılan veri kümesi FracAtlas veri setidir. Bu veri seti, çeşitli radyolojik görüntülerden oluşur. Veri kümesi, farklı tipteki kemik kırıklarını içerir ve etiketlenmiş veri noktalarıyla zenginleştirilmiştir.


VERİ SETİ:https://github.com/VAC10/FracAtlas

![Ekran Görüntüsü (327)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/3ca943dc-4a72-4837-9381-24147dd3d519)
|      | **Fractured** | **Non_Fractured** | **Total** |
|--------------|:----------:|--------------:|----------:|
| **Sınıf Dağılımları**    |    717    |  3366 |      4083 |



# AutoAugmentation
Yukarıdaki sınıf dağılımlarında görüleceği üzere unbalanced(dengesiz) iki sınıfımız mevcut bu iki sınıftan fracture sınıfımızı train,test,valid şeklinde ayırdıktan sonra fracture sınıfının training kısmını non_fracture sınıfının training kısmına eşitledik. Bunu yaparken artırım yapılacak veri üzerinde diğer politikalara göre daha farklı özellikler kullanan ve daha farklı görüntüler elde etmemizi sağlayan  "Autoaugmentation" politikasını tercih ettim.

# Algoritma ve çalışma Mantığı
Algoritma olarak Derin Öğrenmenin güçlü  mimarilerinden olan
MobileNet,DenseNet,Xception mimarileriyle çalıştım. Model çıktılarında bu modellerle ilgili daha geniş bilgiler verilecektir.
Bu projenin çalışma mantığı ve alacağımız çıktılar aşağıda belirtilmiştir.

![Ekran Görüntüsü (357)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/96e82260-e3ba-4ecc-9c6f-aa32dc637c9e)


# Kullanılan Mimariler


## MobileNet
MobileNet,Google tarafından geliştirilen hafif ve etkili bir evrişimli sinir ağı (CNN) mimarisidir. Mobil cihazlar gibi kaynak kısıtlı ortamlarda kullanılmak üzere tasarlanmıştır. Bu mimari, yüksek doğrulukla birlikte düşük hesaplama gücü ve bellek gereksinimleriyle öne çıkar. Özellikle nesne tanıma, yüz tanıma ve diğer görüntü sınıflandırma görevlerinde yaygın olarak kullanılır. MobileNet'in temel amacı, büyük ve karmaşık ağların (örneğin, VGG, ResNet) hesaplama ve bellek gereksinimlerini azaltmaktır. Bu amaçla, bir dizi özelleştirilmiş mimari değişikliği ve optimizasyon yapılmıştır.

![MobileNet-Arch](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/4acd9417-a9a6-4da7-a6e8-07579f3b1de4)


## DenseNet
DenseNet, yoğun bağlantıları ve yoğun bloklarıyla bilinen derin öğrenme modelidir. ResNet'in bir tür genelleştirmesi olarak düşünülebilir. Yoğun bağlantılar, her katmanın önceki tüm katmanlarla birleştirilmesini içerir, bu da daha fazla gradyan akışını ve daha iyi öğrenme hızını sağlar. Bu, ağın "büyümesi" anlamına gelir, çünkü her katmanın önceki katmanların çıktılarına erişimi vardır.

![DenseNet-Arch](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/f947509c-087e-442b-a75c-cb3d0151f348)

## Xception
Xception, Extreme Inception olarak da adlandırılan bir evrişimli sinir ağı (CNN) mimarisidir. Bu mimari, Google tarafından geliştirilmiştir ve ImageNet veri seti üzerindeki görevlerde başarıyla kullanılmıştır. Xception, hesaplama maliyetini azaltmak ve modelin öğrenme kapasitesini artırmak için bir dizi yenilikçi teknik kullanır.

![Xception-architecture](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/facfb069-0a3d-45dc-a9b8-c7726fe9c5ef)



