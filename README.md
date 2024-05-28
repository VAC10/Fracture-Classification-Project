# DERİN ÖĞRENME TABANLI X-RAY GÖRÜNTÜLERİNDEN KIRIK KEMİK SINIFLADIRMA 
Bu proje, derin öğrenme yöntemlerini kullanarak radyolojik görüntülerdeki kemik kırıklarını sınıflandırmak için geliştirilmiştir. Kemik kırıklarının doğru bir şekilde sınıflandırılması, tıbbi teşhis süreçlerinde önemli bir rol oynayabilir, Hızlı ve güvenilir tanı için destek sağlar ve tedavi planlaması için değerli bilgiler verebilir.



# AMAÇ
Bu projenin ana amacı, derin öğrenme ve görüntü işleme tekniklerini kullanarak radyolojik görüntülerdeki kemik kırıklarını otomatik olarak sınıflandırmaktır. Bu sınıflandırma işlemi, farklı kemik kırığı tiplerinin tanımlanmasını ve teşhis edilmesini kolaylaştırarak sağlık profesyonellerine değerli bir araç sunabilir.

# KULLANILAN TEKNOLOJİLER
Python programlama dili kullanılıp; Sklearn, pandas, matplotlib, keras, tensorflow, numpy,seaborn vs. Kütüphaneler kullanılmıştır.
Bu model, Kaggle'a Nvidia tarafından sunulan, yüksek performanslı GPU "P100" ile eğitilmiştir.

# VERİ KÜMESİ
Bu projede kullanılan veri kümesi FracAtlas veri setidir. Bu veri seti, çeşitli radyolojik görüntülerden oluşur. Veri kümesi, farklı tipteki kemik kırıklarını içerir ve etiketlenmiş veri noktalarıyla zenginleştirilmiştir.


VERİ SETİ:https://github.com/VAC10/FracAtlas

![Ekran Görüntüsü (327)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/3ca943dc-4a72-4837-9381-24147dd3d519)
|      | **Fractured** | **Non_Fractured** | **Total** |
|--------------|:----------:|--------------:|----------:|
| **Sınıf Dağılımları**    |    717    |  3366 |      4083 |



# AUTOAUGMENTATİON
Yukarıdaki sınıf dağılımlarında görüleceği üzere unbalanced(dengesiz) iki sınıfımız mevcut bu iki sınıftan fracture sınıfımızı train,test,valid şeklinde ayırdıktan sonra fracture sınıfının training kısmını non_fracture sınıfının training kısmına eşitledik. Bunu yaparken artırım yapılacak veri üzerinde diğer politikalara göre daha farklı özellikler kullanan ve daha farklı görüntüler elde etmemizi sağlayan  "Autoaugmentation" politikasını tercih ettim.

# ALGORİTMA VE ÇALIŞMA MANTIĞI
Algoritma olarak Derin Öğrenmenin güçlü  mimarilerinden olan
MobileNet,DenseNet,Xception mimarileriyle çalıştım. Model çıktılarında bu modellerle 
ilgili daha geniş bilgiler verilecektir.
Bu projenin çalışma mantığı ve alacağımız çıktılar aşağıda belirtilmiştir.
i![Algorithm](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/fc9cea12-0542-4da7-b19a-e7b1c9682914)


# KULLANILAN MİMARİLER


## MobileNet
* MobileNet,Google tarafından geliştirilen hafif ve etkili bir evrişimli sinir ağı (CNN) mimarisidir. Mobil cihazlar gibi kaynak kısıtlı ortamlarda kullanılmak üzere tasarlanmıştır. Bu mimari, yüksek doğrulukla birlikte düşük hesaplama gücü ve bellek gereksinimleriyle öne çıkar. Özellikle nesne tanıma, yüz tanıma ve diğer görüntü sınıflandırma görevlerinde yaygın olarak kullanılır. MobileNet'in temel amacı, büyük ve karmaşık ağların (örneğin, VGG, ResNet) hesaplama ve bellek gereksinimlerini azaltmaktır. Bu amaçla, bir dizi özelleştirilmiş mimari değişikliği ve optimizasyon yapılmıştır.

### MobileNet Mimarisi:
![MobileNet-Arch](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/4acd9417-a9a6-4da7-a6e8-07579f3b1de4)


## DenseNet
* DenseNet, yoğun bağlantıları ve yoğun bloklarıyla bilinen derin öğrenme modelidir. ResNet'in bir tür genelleştirmesi olarak düşünülebilir. Yoğun bağlantılar, her katmanın önceki tüm katmanlarla birleştirilmesini içerir, bu da daha fazla gradyan akışını ve daha iyi öğrenme hızını sağlar. Bu, ağın "büyümesi" anlamına gelir, çünkü her katmanın önceki katmanların çıktılarına erişimi vardır.

### DenseNet Mimarisi:
![DenseNet-Arch](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/f947509c-087e-442b-a75c-cb3d0151f348)

## Xception
* Xception, Extreme Inception olarak da adlandırılan bir evrişimli sinir ağı (CNN) mimarisidir. Bu mimari, Google tarafından geliştirilmiştir ve ImageNet veri seti üzerindeki görevlerde başarıyla kullanılmıştır. Xception, hesaplama maliyetini azaltmak ve modelin öğrenme kapasitesini artırmak için bir dizi yenilikçi teknik kullanır.

### Xception Mimarisi:
![Xception-architecture](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/facfb069-0a3d-45dc-a9b8-c7726fe9c5ef)

# Vision Transformer
Vision Transformers (ViTs), bilgisayarla görme görevlerinde, özellikle görüntü sınıflandırma, nesne algılama ve segmentasyon gibi görevlerde oldukça başarılı olan, Transformer mimarisine dayanan bir derin öğrenme modelidir.
ViT, klasik konvolüsyonel sinir ağları (CNN) yerine, görüntüleri işlemede saf Transformer mimarisini kullanır. Transformer'lar, NLP'de yaygın olarak kullanılır ve kendilerine özgü "self-attention" mekanizması sayesinde uzun menzilli bağımlılıkları iyi bir şekilde modelleyebilirler.

## ViT'lerin Temel Bileşenleri

### Girdi Bölme (Patch Embedding):
* ViT'lerde, bir giriş görüntüsü sabit boyutlu kare yamalara (patch) bölünür. Örneğin, 224x224 boyutunda bir görüntü 16x16 boyutlu yamalara bölünebilir, bu da 196 adet (14x14) yama elde edilmesini sağlar.
Her bir yama düzleştirilir ve bir doğrusal katman ile daha yüksek boyutlu bir vektöre (embedding) dönüştürülür.

### Pozisyonel Kodlama (Positional Encoding):

 * Transformer mimarisi, dizilim içindeki konum bilgilerini kullanmaz. Bu nedenle, yama dizisine konumsal bilgi eklemek için pozisyonel kodlama kullanılır. Bu kodlar, yamaların dizilimdeki konumlarını belirtir ve modelin dizideki konum bilgilerini öğrenmesini sağlar.

### Transformer Blokları:
 * ViT, bir dizi Transformer bloğundan oluşur. Her bir blok, bir multi-head self-attention katmanı ve takip eden iki tam bağlantılı (feed-forward) katman içerir.
Self-attention mekanizması, yamalar arasındaki uzun menzilli bağımlılıkları modelleyerek, her yamanın diğer yamalarla olan ilişkilerini öğrenir.

### Sınıflandırma Başlığı(Classification Head):
* Transformer bloklarından gelen çıktılar, sınıflandırma görevine uygun olarak işlenir. Bu genellikle, özel bir sınıflandırma token'ı ekleyerek yapılır.
Bu token, son Transformer bloğundan çıkan özellikleri kullanarak, bir sınıflandırma katmanına beslenir ve nihai sınıflandırma sonuçları elde edilir.

### ViT Mimarisi:
![Ekran Görüntüsü (398)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/9d75225f-e8fa-4b8f-90d9-b69ebf11fa3a)


# ENSEMBLE LEARNİNG
* Denenen Mimarilerden "Transfer Learning" ile mevcut üç mimarinin(MobileNet,DenseNet,Vit) Feature extraction ile modelin son 
  konvolüsyonel katmanlarından elde edilen yüksek seviyeli derin özellikler elde edilip, vektörize edilir ve başka bir Neural Network'e verilir.


### Vektörize Etme İşlemi:
Vektörleştirme işlemi, ham veriyi (metin, görüntü, ses vb.) makine öğrenimi modellerinin işleyebileceği sayısal formatlara dönüştürme sürecidir. 
* Herbir modelin son katmanından elde edilen High Level Deep Feature(Yüksek seviyeli derin özellikler) aşağıdaki gibi vektörize edilirler.
  
![vektörize işlemi](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/f9fcc9e2-e7ec-4efe-a4dc-80956dd8b722)


  
  ### ENSEMBLE MİMARİSİ:
![Ekran Görüntüsü (397)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/235ae0b8-2f1a-49ae-8a34-e2eec339f6bd)

![Ekran Görüntüsü (404)](https://github.com/VAC10/Fracture-Classification-Project/assets/81007065/9ea02de3-3af7-4b09-bd34-d7c72a3213e7)






