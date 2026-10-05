# Telemetry Signal Anomaly Detection using Autoencoders

Bu proje, savunma sanayii ve otonom sistemlerde kullanılan çok kanallı sensör ve telemetri sinyallerindeki gürültü ve anomalilerin tespiti için geliştirilmiş PyTorch tabanlı bir Autoencoder modelidir.

## Problem Tanımı
Sensör verilerinde etiketli anomali verisi bulmak pratikte zordur. Bu nedenle denetimsiz öğrenme (unsupervised learning) yaklaşımı kullanılmıştır. Model, nominal (normal) operasyonel veriler üzerinde eğitilerek düşük boyutlu uzaya (latent space) sıkıştırılır ve ardından yeniden oluşturulur (reconstruction). Yeniden oluşturma hatası (Mean Squared Error) istatistiksel eşiği aştığında anomali alarmı üretilir.

## Yöntem ve Matematiksel Altyapı
- **Boyut İndirgeme:** 4 kanallı telemetri verisi 2 boyutlu bir manifolda sıkıştırılır.
- **Kayıp Fonksiyonu:** Ortalama Kare Hata (MSE Loss) kullanılarak sinyal rekonstrüksiyon doğruluğu maksimize edilir.
- **Eşik Belirleme:** Rekonstrüksiyon hatasının nominal dağılımı üzerinden istatistiksel $\mu + 3\sigma$ eşikleme kriteri uygulanmıştır.

## Kurulum ve Çalıştırma
Gerekli kütüphaneleri yükleyin ve modeli çalıştırın:

```bash
pip install torch numpy pandas scikit-learn
python model_and_pipeline.py
## Calibration Notes
Reconstruction error threshold set using 99th percentile of validation loss. Evaluating Gaussian KDE for dynamic multi-modal thresholding.
