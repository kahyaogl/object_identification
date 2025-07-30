import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io


import numpy as np

def simulate_vibration(animal_type, duration=1.0, fs=1000, distance=1.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Mesafeye bağlı zayıflama katsayısı (örnek: 1 / d^1.5)
    attenuation = 1 / (distance ** 1.5)

    if animal_type == "insan":
        base_signal = 1.0 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(len(t))
    elif animal_type == "arac":
        base_signal = 2.0 * np.sin(2 * np.pi * 5 * t)
        noise = 0.3 * np.random.randn(len(t))
    elif animal_type == "hayvan":
        base_signal = 0.9 * np.sin(2 * np.pi * 15 * t)
        noise = 0.1 * np.random.randn(len(t))
    else:
        base_signal = np.zeros(len(t))
        noise = 0.1 * np.random.randn(len(t))

    # Mesafeye göre zayıflamış toplam sinyal
    signal = attenuation * base_signal + noise

    return signal, fs





def save_spectrogram(sig, fs, label, index, output_dir="spectrograms"):
    f, t, Sxx = spectrogram(sig, fs=fs)
    
    # Görseli oluştur
    plt.figure(figsize=(2, 2))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.axis('off')  # eksenleri kapat
    
    # Klasörleri oluştur
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Dosyayı kaydet
    file_path = os.path.join(label_dir, f"{label}_{index}.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

for i in range(50):  # her hayvandan 50 örnek
    sig, fs = simulate_vibration("insan")
    save_spectrogram(sig, fs, label="insan", index=i)

    sig, fs = simulate_vibration("arac")
    save_spectrogram(sig, fs, label="arac", index=i)

    sig, fs = simulate_vibration("hayvan")
    save_spectrogram(sig, fs, label="hayvan", index=i)

""""spectrograms/
├── kedi/
│   ├── kedi_0.png
│   ├── kedi_1.png
│   └── ...
├── köpek/
│   ├── köpek_0.png
│   └── ...
└── kuş/
    ├── kuş_0.png
    └── ...
"""

image_size = (64, 64)  # küçük boyutta eğitmek daha hızlı olur
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2  # %80 eğitim, %20 doğrulama
)

train_generator = train_datagen.flow_from_directory(
    'spectrograms',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'spectrograms',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # kuş, kedi, köpek gibi sınıflar
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, 
          validation_data=val_generator,
          epochs=10)


#TEK SİNYALDE SPEKTGROM GÖRSELİ OLUŞTUR
def generate_spectrogram_image(signal, fs, image_size=(64, 64)):
    from scipy.signal import spectrogram
    import matplotlib.pyplot as plt

    f, t, Sxx = spectrogram(signal, fs=fs)
    
    fig = plt.figure(figsize=(2, 2))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)#bellekte geçici tutar işlem hızlılığı sağlar

    image = Image.open(buf).convert("RGB").resize(image_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)  # CNN'in beklediği şekil
#MODEL TAHMİNİ
def predict_animal(signal, fs, model, class_indices):
    img = generate_spectrogram_image(signal, fs)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    # class_indices tersine çevrilir (index → etiket)
    label_map = {v: k for k, v in class_indices.items()}
    return label_map[predicted_class], prediction[0][predicted_class]

# Yeni bir sinyal simüle et 
sig, fs = simulate_vibration("hayvan")

# Tahmin et
class_name, confidence = predict_animal(sig, fs, model, train_generator.class_indices)

print(f"Tahmin edilen hayvan: {class_name} ({confidence*100:.2f}%)")

