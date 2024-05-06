from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

# Test veri seti için tahminler yapma
y_pred = model_densenet.predict(X_test)
y_pred = np.round(y_pred).reshape(-1)  # Sigmoid çıkışı 0 veya 1'e dönüştürülür

# Confusion matrix oluşturma
cm = confusion_matrix(y_test, y_pred)

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, cmap='Blues', fmt='.2%')  # Yüzde formatında göster
plt.title('Confusion Matrix (%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
