# Kullanıcıdan giriş verisini al
reg_value = float(input("Lütfen reg değerini girin: "))
power_value = float(input("Lütfen power değerini girin:"))

# Kullanıcının girdiği değerleri eğitim verileri üzerinde yapılan ölçeklendirmeye tabi tutma
input_data = scaler.transform(np.array([[0, reg_value, power_value, 0]]))

# Daha önce belirlenen pencere boyutu kadar veri al
input_sequence = data_scaled[-window_size:, :]

# Kullanıcının girdiği değeri pencereye ekle
input_sequence[:, 1:3] = input_data[:, 1:3]

# Tahmin yapma
input_sequence = input_sequence.reshape((1, window_size, input_sequence.shape[1]))
predicted_yukh = model.predict(input_sequence)

# Tahmin edilen 'yukh' değerini ölçeklendirmeyi tersine çevirme
predicted_yukh = scaler.inverse_transform(np.array([[0, predicted_yukh[0, 0], 0, 0]]))[0, 1]

print(f"Tahmin edilen yukh değeri: {predicted_yukh}")
