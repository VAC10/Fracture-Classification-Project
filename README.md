# X-Ray görüntülerinden kırık kemik sınıflandırması
Bu proje, derin öğrenme yöntemlerini kullanarak radyolojik görüntülerdeki kemik kırıklarını sınıflandırmak için geliştirilmiştir. Kemik kırıklarının doğru bir şekilde sınıflandırılması, tıbbi teşhis süreçlerinde önemli bir rol oynayabilir ve tedavi planlaması için değerli bilgiler sağlayabilir.



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
