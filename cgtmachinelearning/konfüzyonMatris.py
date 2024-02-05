from sklearn.metrics import confusion_matrix

# Tahminleri sınıflara dönüştürme (örneğin, eşik değeri 0.5 kullanarak)
predicted_classes = (test_predictions > 0.5).astype(int)

# Gerçek sınıfları almak için eşik değeri belirleyin (örneğin, 0.5)
true_classes = (y_test > 0.5).astype(int)

# Konfüzyon matrisini oluşturun
conf_matrix = confusion_matrix(true_classes, predicted_classes)

print("Confusion Matrix:")
print(conf_matrix)
