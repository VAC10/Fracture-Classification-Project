import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Model tahminlerini alalım
y_pred = model_densenet.predict(X_test)

# ROC eğrisi için false positive oranı (fpr) ve true positive oranı (tpr) hesaplayalım
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# ROC eğrisi altındaki alanı (AUC) hesaplayalım
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizelim
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
