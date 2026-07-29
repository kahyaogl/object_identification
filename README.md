# Nesne Tanıma (Spektrogram Tabanlı)

Ses/sinyal verisinden üretilen **spektrogramlar** üzerinden nesne/olay tanıma yapan bir uygulama.

## Özellikler

- **Nesne tanıma modülü** (`nesneTanıma.py`): Spektrogram görüntüleri üzerinden sınıflandırma/tanıma işlemi.
- **Uygulama** (`uygulama.py`): Modelin çalıştırıldığı ana giriş noktası.
- **Spektrogramlar** (`spectrograms/`): Ses/sinyal verisinin görüntüye dönüştürülmüş hali.

## Kullanılan Teknolojiler

- Python
- Sinyal işleme (ses → spektrogram dönüşümü)
- Görüntü tabanlı sınıflandırma

## Kurulum

```bash
git clone https://github.com/kahyaogl/object_identification.git
cd object_identification
pip install numpy librosa matplotlib scikit-learn tensorflow   # gerçek bağımlılıklarını kendi bilgine göre düzenle
python uygulama.py
```

## Kullanım

`uygulama.py` çalıştırıldığında, `spectrograms/` klasöründeki veya girdi olarak verilen ses/sinyal verisinden spektrogram üretilir ve `nesneTanıma.py` içindeki model ile sınıflandırılır.

## Notlar

> Bu README, repo dosya yapısına göre hazırlanmıştır. Kullanılan veri kaynağı (ses/sensör), spektrogram üretim yöntemi ve model mimarisi gibi detayları kendi bilgine göre tamamla.
