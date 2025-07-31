import numpy as np
import matplotlib.pyplot as plt#spektrogramları görüntüye çevirmek için
import io#Görüntüyü bellekte (RAM'de) geçici olarak tutmak için
from tensorflow.keras.preprocessing.image import img_to_array#Görüntüyü modele girecek şekilde array’e çevirir.
from PIL import Image#görüntüleri işlemek için 
import random
from scipy.signal import spectrogram#zaman ve frekans analizi için spektgrom oluşturur
import tensorflow as tf#derin öğrenme katmanları oluşturmak için 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

# --- Ayarlar ---
labels = ["insan", "arac", "hayvan"]#sınıflar
image_size = (64, 64)#cnn için üretilen görüntü boyutu
batch_size = 16#eğitimde her seferde işlenecek örnek sayısı
epochs = 10#eğitimdeki tekrar sayısı

# --- Fonksiyonlar ---

def generate_spectrogram_image(signal, fs, image_size=image_size):
    """
    Verilen titreşim sinyalinden spektrogram görüntüsü oluşturur.
    """
    f, t, Sxx = spectrogram(signal, fs=fs)
    #spektrogramı daha net hale getirip eksenleri siler
    fig = plt.figure(figsize=(2, 2))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.axis('off')
    #görüntüyü ram de saklar dosyaya yazmaz
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    #görüntü açılır 64x64 boyutuna getirir ve normalize eder
    image = Image.open(buf).convert("RGB").resize(image_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)#modele girmesi için boyut ekleme

def simulate_vibration(type, duration=1.0, fs=1000, distance=1.0):
    """
    Hayvan türüne ve mesafeye bağlı titreşim sinyali simülasyonu.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)#zaman vektörü
    attenuation = 1 / (distance ** 1.5)#mesafe artıkça zayıflama efekti

    if  type == "insan":#insan için 20 hz sinyal üretir üzerine gürültü eklenir
        base_signal = 1.0 * np.sin(2 * np.pi * 20 * t)
        noise = 0.2 * np.random.randn(len(t))
    elif type == "arac":#araç için 5 hz daha güüçlü sinyal
        base_signal = 2.0 * np.sin(2 * np.pi * 5 * t)
        noise = 0.3 * np.random.randn(len(t))
    elif type == "hayvan":#hayvan için 15
        base_signal = 0.9 * np.sin(2 * np.pi * 15 * t)
        noise = 0.1 * np.random.randn(len(t))
    else:
        base_signal = np.zeros(len(t))
        noise = 0.1 * np.random.randn(len(t))

    signal = attenuation * base_signal + noise#zayıflatılmış sinyal ve gürültü toplar
    return signal, fs

def get_distance_class(distance):
    """
    Mesafeyi sınıflara ayırır.
    """
    if distance < 3:
        return 0
    elif distance < 10:
        return 1
    elif distance < 30:
        return 2
    else:
        return 3

# --- Veri Üretimi ---

X = []
y_class = []
y_distance = []

for i in range(50):  # Her türden 50 örnek
    for label in labels:
        distance = random.uniform(0.5, 50)#0.5 ile 50 metre arası
        sig, fs = simulate_vibration(label, distance=distance)#sinyali simüla etme
        img = generate_spectrogram_image(sig, fs)#spektrogram oluşturur
        X.append(img[0])  # görüntü listeye eklenir 
        y_class.append(labels.index(label))#etiketler ayrı kaydedilir
        y_distance.append(get_distance_class(distance))

X = np.array(X) #tüm veriler numpy array'e çevrilir
y_class = to_categorical(np.array(y_class), num_classes=len(labels))#etiketler one-hot encoding e cevrilir
y_distance = to_categorical(np.array(y_distance), num_classes=4)

# --- Model Tanımı ---
#cnn mimarisi : 2 konvülasyon +havuzlama aradından tam bağlı katman
inputs = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
#iki ayrı çıkış katmanı (çok görevli öğrenme) 
output_class = Dense(len(labels), activation='softmax', name='class_output')(x)
output_distance = Dense(4, activation='softmax', name='distance_output')(x)
#model aynı girdiden iki farklı sınflama  yapılcak
model = Model(inputs=inputs, outputs=[output_class, output_distance])
#model derlenir :iki ayrı kayıp ve başarı metriği
model.compile(
    optimizer='adam',
    loss={
        'class_output': 'categorical_crossentropy',
        'distance_output': 'categorical_crossentropy'
    },
    metrics={
        'class_output': 'accuracy',
        'distance_output': 'accuracy'
    }
)

# Model Eğitime başlar 0.2 veri doğrulama için ayrılır

model.fit(X, [y_class, y_distance], epochs=epochs, batch_size=batch_size, validation_split=0.2)

# --- Tahmin Fonksiyonu ---
# yeni sinyal için spektrogram oluşturur model tahmini yapar
def predict_animal_with_distance(signal, fs, model, label_map):
    img = generate_spectrogram_image(signal, fs)
    prediction_class, prediction_distance = model.predict(img)
    #tahmin edilen sınıfların indexlerini alır
    predicted_class_index = np.argmax(prediction_class[0])
    predicted_distance_index = np.argmax(prediction_distance[0])
    #etiketleri stringe çevirir 
    predicted_label = label_map[predicted_class_index]
    distance_labels = ['0.5-3m', '3-10m', '10-30m', '30-50m']
    predicted_distance_label = distance_labels[predicted_distance_index]
    
    confidence = prediction_class[0][predicted_class_index]
    # Eğer güven %80'in altındaysa diğer olasılıkları da listele
    other_probs = {}
    if confidence < 0.8:
        for i, prob in enumerate(prediction_class[0]):
            if i != predicted_class_index:
                other_probs[label_map[i]] = prob
    return predicted_label, confidence, predicted_distance_label,other_probs

# --- Örnek Kullanım ---

label_map = {i: label for i, label in enumerate(labels)}

sig, fs = simulate_vibration("hayvan", distance=7.5)
predicted_label, confidence, predicted_distance_label, other_probs = predict_animal_with_distance(sig, fs, model, label_map)

print(f"Tahmin edilen tür: {predicted_label} (%{confidence*100:.2f})")
print(f"Tahmin edilen mesafe aralığı: {predicted_distance_label}")

if confidence < 0.8:
    print("\nDüşük güven nedeniyle diğer olasılıklar:")
    for label, prob in other_probs.items():
        print(f"- {label}: %{prob*100:.2f}")
else:
    print("\nGüven yüksek, diğer tahminler listelenmedi.")





