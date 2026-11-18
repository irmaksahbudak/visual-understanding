# Visual Understanding

Bu proje tek bir RGB robot demonstrasyonundan nesne ve kavrayıcı
yörüngeleri ile görev fazlarını çıkarır.

- SAM 2 nesne maskesini video boyunca takip eder.
- CoTracker3 kavrayıcı üzerinde seçilen noktaları takip eder.
- Kural tabanlı FSM, hareket ölçümlerinden görev fazlarını tahmin eder.
- `pipeline/process_video.py` ana akışı, CSV ve görsel çıktıları üretir.

## Çalıştırma

Ana işlev `pipeline.process_video.process_video`. Ham video,
SAM 2.1 Tiny ağırlıkları, başlangıç maskesi, maskenin referans karesi
ve kavrayıcı başlangıç noktaları girdi olarak verilir. Başlangıç
maskesi ve noktalar bu sürümde elle seçilir. İlk deney Colab'da,
GPU ile çalıştırıldı. Ham video ve model ağırlıkları repoda yoktur.

## ArUco entegrasyonu

Örnek kareler ve FSM CSV'si ayrı paylaşım paketindedir. ArUco
çıktısını aynı kare dizisinde üretin ve `frame` üzerinden eşleştirin.
Kare dizisi 10 FPS'tir. ArUco modülü henüz eklenmemiştir.

## Doğrulama

Modüler FSM, geliştirme videosundaki önceki çıktıyı 277/277 karede
yeniden üretti. Bu sayı yeni videolardaki görev tanıma doğruluğu değildir.
