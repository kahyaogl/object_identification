import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential#cnn modeli
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout#cnn katmanları
from tensorflow.keras.optimizers import Adam
import cv2

# === AŞAMA 1: Sinyal Simülasyonu ===
def generate_signal(freq, noise_level, duration=1.0, fs=1000):#0 ile duration arası eşit aralıklarla zaman noktası üret
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.sin(2 * np.pi * freq * t)#ferkanslı sinüs dalgası üret
    signal += noise_level * np.random.randn(len(t))#gürültülü sinyal
    return t, signal #fonksiyon zaman ve gürültü sinyalini döndürür

def simulate_signals():
    classes = {
        "metal": (120, 0.1),#metal sınıfı için frekans ve gürültü seviyesi
        "plastik": (60, 0.3),#cam sınıfı için frekans ve gürültü seviyesi
        "ahsap": (30, 0.2),#ahsap sınıfı için frekans ve gürültü seviyesi
    }

    os.makedirs("sim_data", exist_ok=True)# sim data klasörü oluştu
    for label, (freq, noise) in classes.items():#her class için frekanns ve gürültü seviyesi al
        folder = f"sim_data/{label}"#sim_data/metal gibi klasör olu
        os.makedirs(folder, exist_ok=True)#her sınıfın sinyallerini ayrı klasöre kaydedecek
        for i in range(50):#50 kez döner 
            t, sig = generate_signal(freq, noise)#generate_signal fonksiyonunu çağırır
            np.save(f"{folder}/{i}.npy", sig)#sinyali npy dosyası olarak kaydeder

"""sim_data/ böyle bir yapıya sahip olacak
├── metal/
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── plastik/
│   ├── 0.npy
│   └── ...
└── ahsap/
    ├── 0.npy
    └── ..."""

# === AŞAMA 2: Spektrogram Görsel Üretimi ===
def save_spectrogram(sig, fs, out_path):#oluşturalacak spektrogramı kaydeder
    f, t, Sxx = spectrogram(sig, fs=fs)#f(frekans) x ekseni,t(zaman) y ekesi,sxx=f ve t için enerji yoğunluğu
    plt.figure(figsize=(2, 2))#görsel alanı oluştur 2x2
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')#2d reenkli grafik
    plt.axis('off')#x ve y eksenini kaldırır
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)#boşluklari kaldırarak kaydeder
    plt.close()#kapatır

    

def generate_spectrogram_images(fs=1000):#spektrogram görsellerini oluşturur
    os.makedirs("images", exist_ok=True)#images adlı klasör oluşturma
    for label in os.listdir("sim_data"):#sim data klasörlerini gezer
        input_folder = f"sim_data/{label}"#sinyallerin bulunduğu klasör
        output_folder = f"images/{label}"#spektrogramların kaydedileceği klasör
        os.makedirs(output_folder, exist_ok=True)
        for file in os.listdir(input_folder):#her sinyal dosyasını (.npy)sırasıyla işler
            sig = np.load(f"{input_folder}/{file}")
            save_spectrogram(sig, fs, f"{output_folder}/{file.replace('.npy', '.png')}")#spektrogram görüntülerini kaydeder
"""sim_data/ yapısı bu şekilde oldu
├── normal/
│   ├── 001.npy
│   └── 002.npy
├── anomaly/
│   ├── 001.npy
│   └── 002.npy

images/
├── normal/
│   ├── 001.png
│   └── 002.png
├── anomaly/
│   ├── 001.png
│   └── 002.png
"""
# AŞAMA 3: CNN Modeli Eğitimi 
def train_cnn_model(img_height=64, img_width=64, batch_size=16):#yeniden boyutlandıracağı hedef boyutlar 16 kez işlencek
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)#0 ve 1 arasında normalize et eğitim ve doğrulama ayrıldı 

    train_gen = datagen.flow_from_directory(# %8o lik eğitim verisi hazırlanıyor
       "images",
        target_size=(img_height, img_width),#64x64 piksele boyutlama
        color_mode='rgb',#renli kanal kullanır
        class_mode='categorical',
        batch_size=batch_size,# her seferde 16 görüntü alır
        subset='training'#eğitim verisi
    )
    """images/
├── normal/
│   ├── 001.png
│   ├── 002.png
├── anomaly/
│   ├── 001.png
│   ├── 002.png
    """



    val_gen = datagen.flow_from_directory(# %20 lik veri için doğrulama verisi hazırlıyor
        target_size=(img_height, img_width),#eğitimin her epochunda doğrulama yapmak için kullanılır
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),#görsellerdeki kenar sekil ferkans özellikleri öğrenir
        MaxPooling2D((2, 2)),#boyutları küçültür verimililiği artırır
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),#özellikleri vektöre dönüştürür 2D--> 1D
        Dense(64, activation='relu'),#nöronlar arasında bağlantı kurar
        Dropout(0.3),#0.3 oranında nöronları rastgele kapatır(eğitim sırasında)
        Dense(3, activation='softmax')  # 3 sınıf tahmini 3 çıktı
    ])

    model.compile(optimizer=Adam(0.001),#yani eğitim sırasında nasıl güncelleneceğini belirler
                  loss='categorical_crossentropy',#çok sınıflandırma için uygun kayıp fonksiyonu
                  metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=15)#15 epoch boyunca eğitim yapar
    return model, train_gen.class_indices#eğitilmiş modeli ve sınıf indekslerini döner

# === AŞAMA 4: Test Tahmini ===
def predict_image(model, class_indices, image_path, img_height=64, img_width=64):
    img = cv2.imread(image_path)#görseli opencv ile okur
    img = cv2.resize(img, (img_height, img_width))#boyutunu modele uygun düzeltirr
    img = img.astype("float32") / 255.0#pikseli 0 -1 arasında nrmalize eder
    img = np.expand_dims(img, axis=0)#4 boyutlu hale getir model bunu bekler
    prediction = model.predict(img)#cnn modeli için olasılık tahmin eder
    predicted_class = np.argmax(prediction)#en büyük olasılığa sahip sınıfı alır
    class_names = list(class_indices.keys())#sınıf isimlerini alır
    print(f"Tahmin: {class_names[predicted_class]} ({prediction[0][predicted_class]*100:.2f}%)")

# ANA AKIŞ 
if __name__ == "__main__":
    print("🔄 1. Sinyal simülasyonu başlıyor...")
    simulate_signals()#sinyal üretir

    print("🖼️ 2. Spektrogramlar oluşturuluyor...")
    generate_spectrogram_images()#sinyali png spektrograma dönüştürür

    print("🧠 3. CNN eğitiliyor...")
    model, class_indices = train_cnn_model()#cnn eğitimi

    print("🔍 4. Test tahmini örneği:")
    # Örnek bir test görüntüsü (ahsap/10.png gibi)
    test_image_path = "images/ahsap/10.png"#test görüntüsü yplu
    predict_image(model, class_indices, test_image_path)
