# Değişiklikler

## 2.4.0: Özel Blok Geliştirme Stüdyosu (Custom Block Studio) & 234 Blok
Toplam blok sayısı **234**'e çıktı (eklentilerle 237+). Kendi özel Python/OpenCV bloklarınızı görsel sihirbazla oluşturma, canlı simüle etme ve tek tıkla palete ekleme yeteneği eklendi.

**Özel Blok Geliştirme Stüdyosu (Custom Block Studio & Wizard)**
- **Görsel Port ve Parametre Yapılandırıcısı:** Blok başlığı, sınıf adı, kategorisi, giriş portları (`IMAGE`, `NUMBER`, `BOOL`, `TEXT`, `CONTOURS`, `DETECTIONS`, `ANY`), çıkış portları ve kullanıcı ayarlarını (int, float, bool, choice, str) görsel tablolar üzerinden yönetebilme.
- **Python / OpenCV Kod Editörü:** Monospace fontlu, 4-boşluk sekme destekli, temiz syntax arayüzlü ve `Node` temel sınıfı ile tam entegre yerleşik kod geliştirme ortamı.
- **Canlı Test & Simülasyon Motoru (▶️ Kodu Test Et):** Kullanıcının yazdığı Python kodunu uygulama içinde sentetik test çerçevesi (640x480 renkli şekiller ve gradyanlar) ve örnek verilerle izole ortamda anında çalıştırma. Çevrim süresini (milisaniye hassasiyetinde), dönen çıktı tiplerini ve değerlerini analiz etme; olası sözdizimi ve çalışma hatalarını satır numarasıyla birlikte renkli hata kutusunda gösterme.
- **6 Hazır Başlangıç Şablonu:**
  1. *🖼️ Görüntü Filtresi (OpenCV / Renk & Morfoloji)*
  2. *🔍 Kusur & Alan Tespiti (Kontur / Eşik Kusur Bulucu)*
  3. *📏 Geometrik Ölçüm & Tolerans (Piksel / mm Mesafe)*
  4. *⚙️ Endüstriyel Karar (OK / NG Kalite Kontrol)*
  5. *🗄️ Metin & Veri Biçimlendirici (Barkod / Seri Port Dizesi)*
  6. *⚡ Boş Özel Blok (Minimal Şablon)*
- **Tek Tıkla Eklenti Kaydı & Dinamik Sıcak Yükleme (Hot-Reload):** `plugins/` klasörüne otomatik Python dosyası oluşturma, çalışma anında modülü yükleyip sol blok paletini anında güncelleme. Kullanıcı yazdığı yeni bloğu programı kapatıp açmadan hemen tuvale sürükleyip kullanabilir.
- **Hızlı Erişim:** Bloklar panelinde doğrudan `🎨 + Özel Blok Oluştur…` butonu ve `Araçlar → 🎨 Özel Blok Geliştirme Stüdyosu… (Ctrl+Shift+B)` menü kısayolu.

**Yeni Eklenen Endüstriyel Bloklar (6 Yeni Blok - Toplam 234)**
- **2 · Görüntü İşleme:**
  - `Kenar Koruyucu Difüzyon (Perona-Malik)`: Kusur kenarlarını ve çatlak sınırlarını keskin tutarken endüstriyel kamera gürültüsünü gideren 2D non-linear difüzyon filtresi.
- **2b · Kalite Kontrol ve Ölçüm:**
  - `Kusur Sınıflandırma Kural Motoru`: Tespit edilen konturları alan, dairesellik ve en/boy oranına göre otomatik olarak Çizik, Çapak, Delik/Gözenek veya Yüzey Lekesi olarak etiketleyip kritiklik kuralı uygulama.
  - `Çizgi Yoğunluk Profili (Caliper Scanner)`: Yatay veya dikey seçilen hat boyunca piksel parlaklık profilini tarayan, 1. türev gradyanla kenar geçişlerini bulan ve mini HUD grafik çizen kumpas tarayıcı.
- **4 · Mantık:**
  - `Parça Varlık / Yokluk Sensörü (P/A)`: Hedef bölgede beklenen parçanın, halkanın, contanın veya kapağın var olup olmadığını alan ve adet toleransıyla denetleyen yüksek hızlı varlık/yokluk mantık bloğu.
- **5 · Çıkış ve Otomasyon:**
  - `Dijital G/Ç Genişletici & Bit Paketleyici`: 8 adet bağımsız lojik durumu tek bir bayt tamsayıya ve Hex dizesine paketleyerek PLC, Modbus ve OPC-UA için durum kelimesi üreten blok.
  - `Üretim Kalite Trend & Iskarta Alarmı`: Hareketli pencere (sliding window) üzerinde anlık verim %, ıskarta oranı ve saatlik üretim hızını izleyen; kalite tolerans dışına çıktığında hat durdurma alarmı üreten blok.

**Kullanıma Hazır Yeni Örnek Eklentiler (Plugins)**
- `plugins/ozel_renk_filtresi.py`: Seçilen HSV/HSL renk aralığını maskeleyen, kontur kutulayan ve piksel sayan örnek blok.
- `plugins/ozel_kusur_sayici.py`: Eşiklemeli kusur sayımı yapan ve izin verilen sınır aşıldığında OK/NG uyarısı veren örnek blok.

## 2.3.2: 228 Blok, Gelişmiş Endüstriyel Metroloji, Lazer Markalama & OEE Analizi
Toplam blok sayısı 190'dan **228**'e çıktı.

**Yeni Arayüz & Görsel Deneyim (UI Modernizasyonu)**
- **Obsidian Dark Teması:** Modern koyu gri tonlar (`#0d1117`, `#161b22`, `#21262d`), yüksek kontrastlı port renkleri, net tipografi (`Segoe UI Variable Display`, `Inter`).
- **Canlı Performans HUD:** Durum çubuğunda anlık FPS renk rozeti (🟢/🟡/🔴), kare işleme gecikmesi (`⚡ ms`), aktif blok sayacı (`🧩 228 Blok`), ve hata sayacı (`⛔`).
- **Önizleme Optik Araç Çubuğu:** Canlı önizleme paneline entegre `✛ Nişangah` (optik merkezleme ve nişan dairesi), `▦ Izgara` (kalibrasyon ve hizalama ızgarası) ve `📷 Kaydet` (görüntüyü tek tıkla masaüstüne PNG kaydetme) düğmeleri.
- **Blok Kartları & Seçim Vurgusu:** 8px yuvarlatılmış köşeler, kategori geçişli renk başlıkları, yumuşak mavi/turuncu seçim ışıması (`QGraphicsDropShadowEffect`).
- **Pencere Simgesi & Araç Çubuğu:** Yüksek çözünürlüklü uygulama logosu, tek tıkla tuvali dışa aktaran "📷 Resim" düğmesi.

**Yeni Eklenen Endüstriyel Bloklar (38 Yeni Blok)**
- **1 · Giriş:**
  - `Ekran / Pencere Yakalama`: Masaüstünü, harici SCADA veya kamera yazılımı pencerelerini canlı video akışı olarak VisionFlow akışlarına aktarma.
- **2 · Görüntü İşleme:**
  - `Mozaik / Gizleme (KVKK)`: Yüz, plaka veya belirlenen alanları sansürleme/bulanıklaştırma.
  - `Filigran / Logo Ekle`: Tarih/saat, logo veya dinamik filigran damgalama.
  - `Otomatik Beyaz Dengesi`: Değişen aydınlatmada gri-dünya renk kalibrasyonu.
  - `Görüntü İyileştir / Sis ve Parlama Gider`: Karanlık kanal ve CLAHE ile endüstriyel sis/parlama giderme.
  - `Termal ve Sahte Renk Paleti`: TURBO, JET, INFERNO, HOT, PLASMA, BONE paletleriyle termal/yoğunluk haritalama.
  - `Parlama ve Yansıma Bastırıcı`: Metalik parçalardaki aşırı ışık parlamalarını (specular highlight) doku onarma (inpaint) veya yumuşak baskı ile giderme.
  - `Otomatik 4 Köşe Perspektif Düzeltici`: Görüntüdeki en belirgin dörtgen parça, etiket veya belgeyi otomatik bularak kuşbakışı dik açıya çevirme.
  - `Lazer / Nokta Vurma Kazıma Vurgulayıcı`: Düşük kontrastlı metal/döküm lazer markalama ve seri numaralarını Sauvola & Niblack adaptif eşikleme ile netleştirme.
  - `Yüzey Çizik ve Çatlak Vurgulayıcı`: Cam, polisajlı metal veya güneş panellerindeki mikro kılcal çizikleri Top-Hat / Black-Hat morfolojik filtrelerle çıkarma.
- **2b · Kalite Kontrol ve Ölçüm:**
  - `Çoklu ROI / Matris Denetleyici`: Blister ambalaj (tablet), şişe kasası veya PCB pin matrisini satır x sütun (NxM) eşzamanlı denetleme.
  - `Kumpas ve Çift Kenar Ölçer (Caliper)`: Çizilen hat boyunca iki zıt kenarı alt-piksel gradyan geçişiyle tespit edip et kalınlığı ve boşluk ölçme.
  - `Barkod & QR Baskı Kalite Kontrolü`: Sembol kontrastı, kenar netliği ve quiet zone analizleriyle ISO/IEC A/B/C/D/F kalite notu üretme.
  - `İstatistiksel Proses Kontrolü (Cp / Cpk / SPC)`: Ortalama, standart sapma, Cp ve Cpk proses yeterlilik indekslerini canlı hesaplama ve tolerans uyarısı verme.
  - `Eşmerkezlilik ve Çap Ölçer`: İç/dış çember çapları, merkez kaçıklığı (offset ΔX, ΔY) ve eşmerkezlilik hatasını sub-piksel ölçme.
  - `Vida ve Cıvata Diş Adımı Kontrolü`: Cıvata/vidaların diş adımı (hatve / pitch), diş sayısı ve derinliğini ölçüp bozuk dişleri tespit etme.
  - `Kusur Maskesi ve Isı Haritası Bindirici`: Kusur maskesini veya AI anomalisini orijinal görüntüye yarı saydam ısı haritası ve dış hatlarla bindirme.
  - `Baskın Renk Bul`: K-Means ile dominant rengi HSV/RGB olarak isimlendirme.
  - `Fark Isı Haritası (Kusur)`: Referans parça ile farkı renklendirilmiş ısı haritası olarak çıkarma.
  - `KPI ve Kalite Panosu (OEE / Verim)`: Toplam, kabul, red, OEE verim ve PPM oranını canlı HUD olarak bindirme.
  - `Akıllı Parça Sınıflandırıcı`: En, boy, alan ve oran kriterlerine göre otomatik kabul/red ve sınıflandırma.
  - `Kusur Detay Büyüteci (PIP)`: Hatalı bölgeyi 2x–5x büyüterek ekran köşesinde resim-içinde-resim (PIP) ve artı işaretiyle gösterme.
  - `Çapak ve Kırık Kenar Analizi`: Parça kenar konturundaki mikro çapakları ve kırıkları piksel hassasiyetinde sayma.
  - `Delta-E Renk Toleransı (CIELAB)`: CIE76 Lab formülüyle renk farkını hesaplama ve tolerans dışı parçayı ayıklama.
- **4 · Mantık:**
  - `PLC / Sistem Canlılık Bekçisi (Watchdog)`: Donanım/kamera darbe sinyalini (heartbeat) denetleme; donma veya gecikmede acil hat durdurma/alarm üretme.
  - `Endüstriyel Veri ve Terazi Çözücü`: Seri port, terazi veya barkod okuyucudan gelen ham ASCII verilerden regex/ayraç ile sayı ve metin ayıklama.
  - `Akıllı Metin Şablonu`: Değişkenleri `{degisken}` biçiminde birleştirerek dinamik rapor/mesaj üretme.
  - `Aykırı Değer Filtresi`: Titreyen veya gürültülü sensör/ölçüm değerlerinden aykırı uç değerleri (outlier) temizleme.
- **5 · Çıkış ve Otomasyon:**
  - `Vardiya OEE & Hat Verimlilik Analizi`: Kullanılabilirlik x Performans x Kalite formülüyle üretim hattı OEE değerini ve parça sayımını gerçek zamanlı çıkarma.
  - `PLC Modbus Word Paketleyici`: 8 farklı lojik sinyali tek bir 16-bit Modbus Holding Register tamsayısına paketleyerek yüksek hızlı haberleşme sağlama.
  - `Özel Frekans Tonu ve Akustik Alarm`: Kabul ve hata durumları için ayrı frekanslarda (ör. 880 Hz / 320 Hz) donanımsal akustik ikaz üretme.
  - `WhatsApp Gönder`: CallMeBot HTTP API ile anlık WhatsApp alarmı.
  - `Google E-Tablolar'a Kaydet`: Webhook ile bulut e-tabloya ölçüm satırı ekleme.
  - `Sesli Uyarı (Bip / WAV)`: Frekans ayarlı sistem bip sesi veya özel WAV ses dosyası çalma.
  - `Windows Sistem Bildirimi (Toast)`: Yerel Windows 10/11 masaüstü toast bildirimi.
  - `Barkod ve QR Kod Damgala`: Code128 ve 2D Data Matrix damgası oluşturup görüntüye işleme.
  - `Sesli Konuşma ve Anons (TTS)`: SAPI ve pyttsx3 ile internetsiz Türkçe sesli konuşma ve alarm anonsu.
  - `Üretim & Kalite CSV Raporlayıcı`: Tarih/saat damgalı, Excel uyumlu günlük UTF-8-BOM CSV üretim günlüğü.

**Paketleme ve Kurulum**
- Tek tıkla bağımsız Windows kurulum paketi: `VisionFlow-Kurulum-Setup.exe`.
- Tüm ek kütüphaneler (pyttsx3, paho-mqtt, pyserial, asyncua, pytesseract) entegre edildi, `--selftest` %100 başarılı.

## 2.3.0: Ekran tasarımcısı, 3B/derinlik, veri seti yöneticisi, istasyon yöneticisi
Toplam blok sayısı 178'den 190'a çıktı.

**Ekran tasarımcısı**
- Yeni: Araçlar → Ekran tasarımcısı… (14 öğe türü, sayfalar, ızgara, hizalama, geri al, önizleme); tasarım `operator.screen` içinde saklanır.
- Yeni: Operatör panelinde "Panel düzeni: Tasarım ekranı"; `--operator` tasarlanan ekranı açar.

**3B ve derinlik**
- Yeni bloklar: Derinlik Kamerası (RealSense, Orbbec, dosya), Derinlik Test Sahnesi, Derinliği Renklendir, Derinlik Filtresi, Zemin Düzlemi Bul, Yükseklik Ölç, Hacim Ölç, 3B Nesne Bul, Düzlemsellik / Eğiklik, Derinlik Profili, Nokta Bulutu Kaydet, 3B Görünüm.
- Yeni: 3B Görüntüleyici penceresi; yeni şablon `examples/derinlik_hacim_olcum.json`.

**Veri setleri**
- Yeni: Araçlar → Veri seti yöneticisi… (istatistik, sorunlar, kopya resim, sınıf işlemleri ve yedek, sınıf dengeli bölme, YOLO/COCO/VOC dönüştürme, veri çoğaltma, otomatik etiketleme ve inceleme).
- Yeni komut: `python -m visionflow veriseti`.
- Düzeltildi: Az Örnekle Nesne Bul kutuları 2–6 px sola-yukarı kayıyordu.

**İstasyonlar ve uzaktan kontrol**
- Yeni: İstasyon yöneticisi (her akış ayrı işlemde, çökeni geri çekilerek yeniden başlatma, durum/önizleme, işlemci önceliği ve çekirdek seçimi); `python -m visionflow istasyonlar`, `servis kur --istasyonlar`.
- Düzeltildi: `--operator` kipinde API açılmıyordu; `servis kur` `--api` alamıyordu; `--api` yeniden başlatma ayarlarını yok sayıyordu; `/durum` önceki akışın reçetesini gösteriyordu.
- Düzeltildi: API'den, ekrandan veya düzenleyiciden seçilen reçete artık panel, tasarlanmış ekran, 'Reçete Seç' blokları ve `/durum` arasında tutarlı; `--operator` → Düzenleyici geçişinde ekranda yazılan ayarlar ve reçete kaybolmuyor.
- Düzeltildi: Aynı çıkışa bağlı görüntü ve değer araçları birbirini bozmuyor; derinlik (float) görüntüleri API JPEG'inde, istasyon önizlemesinde ve izlenebilirlik arşivinde artık bembeyaz/karanlık değil, geçerli değerlere göre geriliyor.

**Doğrulanmayanlar:** Gerçek RealSense/Orbbec kamera, dokunmatik ekran, çoklu monitör ve Windows'ta denenmedi; YOLO ile otomatik etiketleme sahte modelle test edildi.

## 2.2.0: Şablon galerisi, yeni muayene blokları, az örnekle nesne bulma, izlenebilirlik, REST API
Toplam blok sayısı 167'den 178'e çıktı.

**Yeni akış ve şablonlar**
- Yeni: Dosya → Yeni… şablon galerisi (kartlar, akış önizlemesi, kategori, Türkçe harfe duyarsız arama, sorularla ayar doldurma, kaynak bloğunu otomatik değiştirme, ilk açılışta karşılama).
- Değişti: boş akış artık Dosya → Boş akış (Ctrl+Shift+N).
- Yeni: 8 hazır şablon ve örnek resimleri (`examples/ornek_veri/sablonlar/`).

**Muayene blokları**
- Yeni bloklar: Etiket Var mı, Tarih Kodu Doğrula, Renk ile Ayıkla, Kenar Kusuru Bul, Doku / Yüzey Kusuru, Hizalama Kontrolü, Sayaç Karşılaştır.

**Az örnekle nesne bulma**
- Yeni: `visionflow/fewshot_det.py` (dönme/ölçek toleranslı şablon arama + numpy doğrulayıcı, `.vfdet` model dosyası).
- Yeni blok: Az Örnekle Nesne Bul (YOLO tespit biçimi + açı ve döndürülmüş kutu); yeni pencere: Kutu Çiz ve Eğit.
- Yeni örnek: `examples/az_ornekle_nesne_bulma.json`.

**İzlenebilirlik**
- Yeni: `visionflow/trace.py` arşivi (SQLite WAL, gün klasörlerinde JPEG, arka planda yazma, taşmada sayılan atlama, gün/GB saklama, yalnız hatalıların görüntüsü).
- Yeni bloklar: Parçayı Arşivle, Arşivde Ara.
- Yeni pencere: Araçlar → İzlenebilirlik arşivi… ve operatör panelinde Arşiv düğmesi.
- Yeni komut: `python -m visionflow arsiv`.

**Uzaktan kontrol ve güvenlik**
- Yeni: REST API (`/api/v1`: durum, baslat, durdur, degerler, goruntu, tetik, recete, degisken, bloklar, openapi.json); API anahtarı, yalnız-bu-bilgisayar varsayılanı, HTTPS, hız sınırı.
- Yeni: Araçlar → Uzaktan kontrol (API)… penceresi; `--headless --api PORT` (+ `--api-ag`, `--api-sertifika`, `--api-sertifika-anahtari`, `--api-anahtar-goster`, `--api-bekle`).
- Yeni blok: API Tetik.
- Yeni: OPC UA istemci/sunucu bloklarında Basic256Sha256 İmzala / İmzala ve Şifrele, otomatik sertifika (`pki/`), ilk bağlantıda güven veya güvenilen liste.
- Düzeltildi: OPC UA Oku/Yaz bağlanamayınca arka plan iş parçacığı açık kalıyordu.

**Doğrulanmayanlar:** Gerçek kamera, PLC, MES/ERP ve Windows'ta denenmedi. Nesne bulma yalnız sentetik resimlerle ölçüldü; OPC UA güvenliği yalnız VisionFlow ↔ asyncua arasında denendi (Siemens/Kepware değil).

## 2.1.0: OPC UA ve Modbus sunucusu, EtherCAT, Profinet, GPU hızlandırma, az örnekle segmentasyon
Toplam blok sayısı 157'den 167'ye çıktı.

**VisionFlow sunucu olarak**
- Yeni bloklar: OPC UA Sunucusu: Yayınla / Oku (asyncua; paylaşımlı sunucu, isteğe bağlı kullanıcı/şifre).
- Yeni bloklar: Modbus Sunucusu: Yaz / Oku (ek paketsiz Modbus TCP slave; fonksiyon 1, 2, 3, 4, 5, 6, 15, 16; 16/32-bit ve float).
- Yeni örnek: `examples/opcua_sunucu.json`.

**EtherCAT ve Profinet**
- Yeni bloklar: EtherCAT Oku / Yaz / Ağını Tara (pysoem; paylaşımlı master, arka plan çevrimi, OP durum makinesi, kopukluk kurtarma, durunca çıkışları sıfırlama).
- Yeni bloklar: Profinet Cihazlarını Bul (DCP Identify) ve Profinet İstasyon Adı Ata (DCP Set, açık onayla).
- Yeni: süreç verisi için bayt/bit yardımcıları, aralık dışı değerlerde Türkçe hata.
- requirements-extra.txt: pysoem, scapy (isteğe bağlı).

**GPU hızlandırma**
- Yeni: `visionflow/accel.py`: CUDA, TensorRT, OpenVINO, DirectML, CoreML; akış ve bilgisayar düzeyinde varsayılan cihaz, blok başına Cihaz ayarı, FP16, CPU'ya uyarılı geri dönüş.
- Yeni: Araçlar → Hızlandırma (GPU)… penceresi ve hız testi; `python -m visionflow gpu-test`.
- Yeni: ONNX bloklarında `device` çıkışı. Kurulum testi GPU durumunu listeler.

**Az örnekle segmentasyon**
- Yeni: `visionflow/fewshot_seg.py` piksel sınıflandırıcı (renk, çok ölçekli bulanıklık, kenar ve doku öznitelikleri, OpenCV RTrees), `.vfseg` model dosyası.
- Yeni blok: Az Örnekle Segmentasyon; yeni pencere: boyayarak eğitme (fırça, silgi, sınıflar, geri al, canlı önizleme, GrabCut).
- Yeni: blokların kendi penceresini açan "Eğit…" türü düğmeler için ortak altyapı (Ayarlar panelinde).
- Yeni örnek: `examples/az_ornekle_segmentasyon.json`.

- Düzeltildi: Qt'nin kendi düğmeleri (Close, Cancel, Yes/No) ve dosya pencereleri artık Türkçe.

**Doğrulanmayanlar:** Gerçek PLC/SCADA, EtherCAT slave'i, Profinet cihazı, NVIDIA/Intel GPU ve Windows'ta denenmedi. Sunucular yerel istemcilerle, EtherCAT sahte pysoem ile, Profinet elle hazırlanmış paketlerle, GPU seçimi sahte sağlayıcılarla test edildi.

## 2.0.0: Hat ekranı, reçeteler, üretim ve ölçüm blokları, servis, editör araçları
Toplam blok sayısı 137'den 157'ye çıktı.

**Operatör paneli ve rapor**
- Eklendi: Operatör Paneli (HMI) penceresi, `--operator akis.json` komut satırı seçeneği.
- Eklendi: Operatör paneli ayarları akış dosyasında ("operator" anahtarı, yüklemede denetlenir).
- Eklendi: Şifreli düzenleme kilidi (PBKDF2-SHA256, tuzlu özet).
- Eklendi: Dakikalık üretim verisi kaydı (SQLite) ve HTML/PDF üretim raporu (Araçlar → Üretim raporu…).
- Eklendi: "Rapor Oluştur" bloğu.

**Reçeteler ve alt akış ayarları**
- Yeni: Alt akış bloklarında dışarı açılan ayarlar (içerideki bloğun ayarına sağ tık → "Bu ayarı alt akış bloğunda
  göster"); kayıt, kopyala-yapıştır, kütüphane ve geri al ile korunur, çalışırken anında uygulanır.
- Yeni: Reçeteler / ürün tipleri (Araçlar → Reçeteler…): seçili blokların değerlerini reçeteye alma, düzenleme,
  uygulama; reçeteler akış dosyasında saklanır.
- Yeni blok: Reçete Seç — akış çalışırken ada veya numaraya göre ürün tipini değiştirir (kareler arasında güvenli
  uygulama, gerekirse bloğu yeniden başlatma).
- Yeni: `headless --recete AD` ile akışı bir reçeteyle başlatma.

**Ölçüm ve kontrol blokları**
- Yeni: Özellik Eşleştir (Nesne Bul) — dönmeye ve ölçeğe dayanıklı nesne bulma (ORB/SIFT + RANSAC).
- Yeni: Çizgi Bul (Hough) — çizgi sayısı, en uzun çizginin açısı ve uzunluğu.
- Yeni: Kenar Ölç (Kumpas) — piksel altı kenar konumu, px/mm mesafe.
- Yeni: Leke / Kusur Bul (Blob) — yerel arka plan farkıyla leke bulma, alan/yuvarlaklık süzgeci.
- Yeni: Doluluk Seviyesi — sıvı seviyesi % ve eksik/fazla dolum kararı.
- Yeni: Renk Alanı % — HSV aralığındaki alan yüzdesi.
- Yeni: Histogram Karşılaştır — referansla 0..1 renk benzerliği.
- Yeni: Otomatik Hizala — ECC / özellik tabanlı hizalama, dx/dy/açı.
- Yeni: Sayma Izgarası (Tepsi Kontrol) — boş hücreleri bulur; örnek akış tepsi_kontrol.json.

**Üretim hattı blokları**
- Yeni: Ret Kapısı Gecikmesi (parça / enkoder / süre modlu kaydırma yazmacı, ayarlanabilir ret darbesi).
- Yeni: SPC bloğu (X-bar / bireysel kart, Cp, Cpk, UCL/LCL, Western Electric kuralları, öğrenme dönemi).
- Yeni: OEE Hesapla (kullanılabilirlik, performans, kalite, duruş dakikası, vardiya sıfırlama).
- Yeni: Parti / Lot Takibi (elle veya otomatik lot numarası, lot dolunca darbe, kalıcı durum).
- Yeni: Duruş Sebebi Kaydı / Alarm Geçmişi (CSV kaydı, titreşim filtresi, günlük sayı).
- Yeni: Çoklu Kamera Kararı (4 istasyona kadar, sırayla eşleştirme, zaman aşımı alarmı).
- Yeni: Sinyal Üreteci / Simülatör (bantsız deneme için parça, hata, enkoder ve ölçüm üretir).
- Yeni örnek akış: ret_kapisi_simulasyon.json.

**Servis, kurulum, kayıt ve test**
- Yeni: `servis` komutu (Windows servisi / Görev Zamanlayıcı / Linux systemd) ile akışı bilgisayar açılınca çalıştırma.
- Yeni: `--headless ... --yeniden-baslat`: çöken akışı artan beklemeyle yeniden başlatır, dönen kayıt dosyası tutar.
- Yeni: Windows kurulum sihirbazı (Inno Setup, `installer/`), `.json` dosya ilişkilendirme, "Windows ile başlat".
- Yeni blok: Kare Kaydedici (tetikli, öncesi/sonrası tamponlu, disk korumalı kayıt).
- Yeni blok: Kayıttan Oynat (kaydı orijinal zamanlamayla veya hızlı oynatır).
- Yeni: `test-akis` komutu ile kayda karşı akış regresyon testi.
- Değişiklik: `python -m visionflow` komut satırı araçları artık Qt (PySide6) yüklemeden çalışır.

**Editör**
- Eklendi: Hizala / dağıt (Düzen → Hizala, sağ tık menüsü), tek adımda geri alınır.
- Eklendi: Izgaraya yapıştır (20 px, Görünüm menüsü, hatırlanır).
- Eklendi: Mini harita (tuvalin sağ altında, tıkla / sürükle ile gezinme).
- Eklendi: Akışı PNG / SVG resim olarak kaydetme (Dosya menüsü).
- Eklendi: Boş tuvale bırakılan kablo için uyumlu blok listesi ve otomatik bağlama.
- Eklendi: Bloğu bir kablonun üzerine bırakınca araya ekleme (vurgulu).
- Eklendi: Akışta ara (Ctrl+P), bulanık arama ve hata durumu gösterimi.
- Eklendi: Hatalı bloklar arasında gezinme (F8, araç çubuğu düğmesi).
- Eklendi: Açık tema (Görünüm → Tema), tuval renkleriyle birlikte.
- Eklendi: Blok rengi etiketi; akış dosyasına isteğe bağlı "color" alanı olarak kaydedilir (eski sürümler bu alanı yok sayar).


**Düzeltildi (yayından önceki iki hata avında bulundu, 13 hata):** Yapay zeka ile akış değiştirme operatör paneli ayarlarını, düzenleme kilidini ve blok renklerini siliyordu; alt akış yapmak/çözmek reçete değerlerini ve operatör paneli bağlantılarını kaybediyordu; operatör panelinde alt akışların içindeki bloklar seçilemiyordu; F8 sadece önceki blok yüzünden hata veren blokları da geziyordu; operatör paneli alt akış içindeki bir hatayı iki kez sayıyordu; kayıtlı bir sayacın başlangıç değeri (ör. 5000) üretime ekleniyordu; test-akis alt akışların içindeki kayıt bloklarını görmüyordu; servis olarak çalışan akış üretim verisi kaydetmediği için rapor boş geliyordu; reçete alt akışın adını değiştirince başlık güncellenmiyordu, iç içe alt akışlarda canlı reçete yeniden başlatmada eski değere dönüyordu; `--help` editörü açıyordu; çökme sonrası kurtarma kilitli akışı şifre sormadan açıyordu; Esc ile kapatılan reçeteler penceresi arka planda kalıyordu; kütüphaneye kaydedilen alt akış rengini kaybediyordu.

**Testler:** 558 → 767. Tam takım üst üste 3 kez geçti. Gerçek PLC, konveyör, enkoder, endüstriyel kamera ve Windows üzerinde denenmedi; servis ve kurulum sihirbazı yalnızca üretilen komut ve dosyalar üzerinden test edildi.

## 1.9.0: Alt akışlar (grup bloklar)
**Yeni:**
- **Alt akış yap (Ctrl+Alt+G veya sağ tık):** Seçilen bloklar tek bir bloğa dönüşür. Dışarıyla olan bağlantılar o bloğun girişleri ve çıkışları olur; dışarıdaki tek bir çıkış içeride birden çok bloğa gidiyorsa tek giriş açılır. Akış aynı sonucu verir, sadece sadeleşir.
- **İçine gir, düzenle, çık:** Alt akış bloğuna çift tıklayınca içi açılır. Üstte nerede olduğunuz görünür (Ana akış › Alt akış adı); *⬆ Üst akışa dön* (Alt+↑) ile çıkılır. İçeride yapılan her değişiklik geri alınabilir ve akışla birlikte kaydedilir. Alt akış içinde alt akış olabilir.
- **Yeni bloklar (toplam 137):** Alt Akış, Alt Akış Girişi, Alt Akış Çıkışı. Giriş ve çıkış blokları alt akışın portlarıdır: adları port adı, yukarıdan aşağı sıraları port sırasıdır, türleri bağlandıkları yere göre kendiliğinden belirlenir.
- **Çalışırken içi de görünür:** Akış çalışırken bir alt akışın içine girince içerideki blokların önizlemeleri, süreleri ve hataları görünür; ayarlar canlı değiştirilebilir. İçine girip çıkmak akışı yeniden başlatmaz.
- **Alt akış kütüphanesi:** Sağ tık → *Alt akış kütüphanesine kaydet*; başka bir akışta *Blok ekle → Alt akış kütüphanesi*'nden eklenir. *Alt akışı çöz* blokları eski yerine geri koyar.
- **Akış denetimi** alt akışların içine de bakar (ör. "[Kırmızı nesneleri bul] … girişi bağlı değil"). İçerideki bir blok hata verirse alt akış bloğu hangi bloğun hata verdiğini yazar.
- Örnek akış: `examples/alt_akis_nesne_sayma.json`.

**Düzeltildi (yayından önceki hata avında bulundu):** Yapay Zeka ile akışı değiştirmek alt akışların içini boşaltıyordu; çalışırken bir alt akışın içine girip çıkmak akışı yeniden başlatıyordu (kameralar yeniden açılıyor, sayaçlar sıfırlanıyordu); içerideki bir giriş veya çıkışın adını değiştirmek dışarıdaki bağlantıyı sessizce koparıyordu; içeride, alt akışa gelen görüntü önizlemede görünmediği için bölge çizilemiyordu; içerideki her hata "Kendini Geliştir" kaydına iki kez ve yanlış dosyayla yazılıyordu; çok derin iç içe alt akışlar yavaş açılıyordu.

**Testler:** 524 → 558. Tam takım üst üste 3 kez geçti.

## 1.8.0: Milimetre ölçüm, kamera kalibrasyonu, yorum çerçeveleri
**Yeni bloklar (toplam 134):**
- **Lens Bozulmasını Düzelt:** Kamera kalibrasyon dosyasıyla lensin bükmesini düzeltir; düz kenarlar düz olur, resmin kenarındaki ölçüler de doğru çıkar. Farklı en-boy oranındaki bir çözünürlükte açık bir hata verir (yanlış ölçmek yerine).
- **Ölçek Bul (mm):** Görüntüdeki satranç tahtası veya ArUco işaretçisinden "mm başına piksel" bulur. "Kilitle" açıksa bir kez bulduktan sonra referans kaldırılabilir.
- **Kuşbakışı Düzelt (mm):** Açılı bakan kameranın görüntüsünü masadaki satranç tahtasıyla tam üstten bakılmış hale getirir; her yerde 1 mm aynı piksel sayısıdır.
- **Mesafe Ölç:** İki tespitin merkezleri arasındaki mesafe (mm veya piksel); soldan, yukarıdan, en büyük veya en emin iki tespit seçilebilir.
- **Nesne Ölç** artık `px_per_mm` girişi alır: Ölçek Bul veya Kuşbakışı Düzelt çıkışını bağlayınca sonuçlar milimetre olur.

**Araçlar → Kamera Kalibrasyonu:** Yazdırılacak satranç tahtasını kaydeder, 10–20 fotoğraftan lensi kalibre eder (arka planda, pencere donmaz), kaliteyi ve en çok hata veren fotoğrafları gösterir. Farklı çözünürlükteki veya tahtanın bulunmadığı fotoğraflar atlanır ve listelenir.

**Yorum çerçeveleri:** Ctrl+G (veya sağ tık) ile seçili blokların etrafına başlıklı, renkli bir çerçeve koyun. Başlıktan sürükleyince içindeki bloklar da taşınır; köşeden boyutlanır; çift tıkla yeniden adlandırılır; geri al çalışır; akışla birlikte kaydedilir.

**Düzeltildi (yayından önceki hata avında bulundu):** kilitli ölçek, kare boyutu değiştirilince eski değeri vermeye devam ediyordu; Kuşbakışı Düzelt'te köşe sayısı kilitliyken değiştirilince akış çöküyordu; kilit kapalıyken referans görüntüden çıkınca eski ölçek kullanılmaya devam ediyordu; fareyle alan seçip silmek çerçeveleri de siliyordu; Yapay Zeka akış oluşturucusu akışı değiştirince çerçeveler kayboluyordu; sadece çerçeve içeren akış "kaydedilmemiş" sayılmıyordu; bozuk bir dosyada aynı numaralı iki çerçeve karışıklık çıkarıyordu.

**Testler:** 502 → 524. Kalibrasyon sentetik (yazılımla üretilmiş, bilinen lens bozulmalı) bir kamerayla doğrulandı: odak uzaklığı ve bozulma doğru bulundu, 100×50 mm plaka doğru ölçüldü. Gerçek bir kamerayla denenmedi. Tam takım üst üste 3 kez geçti.

## 1.7.0: Poz tahmini, döndürülmüş kutular, derin anomali
**Yeni bloklar (toplam 130):**
- **YOLO Poz Tahmini:** YOLOv8/v11 (ve YOLO26) -pose modelleriyle insan iskeleti: 17 eklem noktası. Kendi eğittiğiniz anahtar nokta modelleri (ör. 4 köşe) de çalışır. Tespitleri Çiz iskeleti çizer.
- **Eklem Açısı:** Üç eklem noktası arasındaki açı (ör. omuz–dirsek–bilek). Ergonomi kontrolü, kol kaldırma, tekrar sayma için.
- **YOLO Döndürülmüş Kutu (OBB):** Yamuk duran parçaları döndürülmüş kutuyla bulur ve açısını (derece) verir. Ultralytics ile aynı yöntemle çakışanları ayıklar; YOLO26 modelleri de çalışır.
- **Derin Anomali (PatchCore):** Sadece 10–50 sağlam ürün resmiyle öğrenir, hiç görmediği çizik, leke ve eksik parçayı işaretler, alarm sınırını kendisi belirler (skor 1'i geçerse hatalı). Yerleşik öznitelikler internetsiz çalışır; ResNet18 seçeneği daha hassastır (bir kez torch gerekir). Eğitim saklanır, resimler değişmedikçe tekrarlanmaz. Hazır örnek: `examples/derin_anomali_patchcore.json`.

**Doğrulama:** Poz ve OBB, gerçek YOLO11 modelleriyle Ultralytics'in kendi sonuçlarıyla karşılaştırıldı: aynı kişiler ve eklemler, gemilerde aynı konum, boyut ve açı.

**Düzeltildi (yayından önceki hata avında bulundu):** YOLO26 OBB modelleri anlamsız sonuç veriyordu; kendi ONNX modelinizi yeniden eğitince eski eğitim kullanılmaya devam ediyordu; aynı klasörü kullanan iki PatchCore bloğu birbirinin eğitimini silip her açılışta yeniden eğitiyordu; çok ve büyük resimle eğitim gigabaytlarca bellek kullanıyordu; yarım kalmış bir eğitim dosyası bloğu kalıcı olarak bozuyordu; saydam (4 kanallı), 16 bit veya ondalıklı görüntüler hata veriyordu; eğitim hazırlığı yavaştı; internet yokken anlaşılmaz bir hata çıkıyordu.

**Testler:** 475 → 502. Tam takım üst üste 3 kez geçti.

## 1.6.0: Endüstriyel kameralar
**Yeni:**
- **Endüstriyel Kamera bloğu (toplam 126):** GigE Vision ve USB3 Vision kameralar. Basler için `pypylon`; Hikrobot, Daheng, FLIR, Allied Vision ve diğer GenICam kameralar için `harvesters` ve üreticinin `.cti` dosyası (program bu dosyayı bilinen kurulum yerlerinde kendisi bulur). Seri numarasıyla kamera seçme, pozlama (µs) ve kazanç (dB) ayarı (akış çalışırken de değişir), Mono/Bayer/RGB ve 10–12 bit görüntüleri renkli görüntüye çevirme. Bağlantı koparsa kendiliğinden yeniden bağlanır.
- **Tetik:** Sürekli, yazılım tetiği (`trigger` girişi açılınca tek kare) ve donanım tetiği (Line0 / Line1: sensör veya PLC sinyaliyle çekim). Tetik beklenirken akış hata vermez ve sonraki bloklar eski kareyi tekrar işlemez, böylece sayımlar şişmez.
- **Araçlar → Endüstriyel Kameraları Bul:** Bilgisayarın gördüğü kameraları listeler; seçtiğinizi tek tuşla akışa ekler. Arama arka planda yapılır, pencere donmaz.

**Düzeltildi (yayından önceki hata avında bulundu):** tetikli çekimde bazı kareler ekranda hiç görünmüyordu; başlayamayan bir bloğun ayarı kaydırılınca kamera her adımda yeniden aranıp akış takılıyordu; tetik beklerken geri besleme bağlantıları yanlış "kapandı" sinyali veriyordu; arayüzsüz çalışmada tetik bekleme turları kare sayılıyordu; kameranın kabul etmediği tetik girişi sessizce sonsuz beklemeye yol açıyordu (artık açık bir hata verir); eski Basler GigE kameralarda kazanç hiç ayarlanmıyordu (artık GainRaw ile ayarlanır); yazılım tetiğinde önceki tetiğin geç gelen karesi yeni kare sanılabiliyordu.

**Testler:** 446 → 475. Sahte Basler ve GenICam kameralarla denendi; gerçek bir endüstriyel kamerayla denenmedi. Tam takım üst üste 3 kez geçti.

## 1.5.0: Daha fazla PLC: Modbus RTU ve Allen-Bradley
**Yeni bloklar (toplam 125):**
- **PLC Oku / PLC Yaz (Modbus RTU):** RS-485 veya RS-232 (USB çevirici ile) üzerinden Modbus RTU cihazlarla konuşur: sıcaklık kontrol cihazları, frekans sürücüleri, enerji analizörleri, eski PLC'ler. Register veya bobin okur/yazar; yazma sadece değer değişince gönderilir. Aynı hattaki birden çok cihaz için birden çok blok koyabilirsiniz; aynı portu paylaşır ve sırayla konuşurlar. CRC, yanlış cihaz, cihaz hata kodu ve cevap gelmemesi anlaşılır Türkçe mesajla bildirilir. Ek paket: `pyserial`.
- **Allen-Bradley Oku / Yaz:** Rockwell ControlLogix, CompactLogix, Micro800, SLC 500 ve MicroLogix PLC'lerle EtherNet/IP üzerinden etiket (tag) okur/yazar. Okuma sayı, açık/kapalı ve yazı olarak verir; yazma değer değişince gönderilir, sayaçtan gelen 3.0 tamsayı etikete 3 olarak yazılır. Ek paket: `pycomm3`.
- README'ye hangi PLC için hangi bloğun kullanılacağını gösteren bir tablo eklendi.

**Düzeltildi (yayından önceki hata avında bulundu):** geç cevap veren bir cihazın eski cevabı bir sonraki okumanın cevabı sanılabiliyordu (hatasız ama yanlış değer); PLC Yaz blokları (Modbus TCP dahil) adres veya tip değiştirilince aynı değeri yeni adrese yazmıyordu; adı F ile başlayan Allen-Bradley etiketlerine (ör. `Fault_Code`) yazma hata veriyordu; akış çalışırken hattın hızı değiştirilince blok başlamıyordu; bağlanamayan Allen-Bradley bloğu PLC'deki bağlantı yerini boşaltmıyordu; port adının sonundaki boşluk portu açtırmıyordu.

**Testler:** 429 → 446. Sahte RS-485 hattı ve sahte PLC ile denendi; gerçek bir PLC veya RS-485 donanımıyla denenmedi. Tam takım üst üste 3 kez geçti.

## 1.4.0: Etiketlemede otomatik kutu önerisi
**Yeni (Araçlar → Etiketleme Aracı):**
- **Model seç + Modelle öner (M):** Seçtiğiniz YOLO ONNX modeli (ör. ilk birkaç yüz resimle eğittiğiniz model) o resimdeki nesneleri kutular. Modelin bildiği ama veri setinizde olmayan sınıflar otomatik eklenir, zaten çizdiğiniz kutular tekrar eklenmez.
- **Etiketsizleri modelle etiketle:** Henüz etiketi olmayan tüm resimleri sırayla kutular; etiketli resimlere dokunmaz, Durdur ile kesilebilir. Siz sadece yanlışları silip eksikleri çizersiniz.
- **Önceki resmin kutuları (C):** Video kareleri veya yerinden oynamayan ürün için; resim boyutu farklıysa kutular ölçeklenir.
- **Ctrl+Z:** Son çizilen kutuyu geri alır.

**Düzeltildi (yayından önceki hata avında bulundu):** toplu etiketleme sürerken çizilen kutuların üzerine yazılabiliyor ve başka klasör açılırsa etiketler yanlış klasöre gidebiliyordu; "parca #2" gibi sınıf adları kesiliyordu; satır sonu veya fazladan boşluk içeren sınıf adları sınıfları kaydırıyor ya da çoğaltıyordu; resmin dışında kalan model kutuları "önerildi" sayılıyordu; aynı adlı a.jpg/a.png bir resmin etiketini siliyordu.

**Testler:** 416 → 429. Gerçek bir ONNX modeliyle de denendi. Tam takım üst üste 3 kez geçti.

## 1.3.0: Kendi kendine öğrenen görüntü
**Yeni:**
- **Emin Olmadıklarını Topla bloğu (toplam 121):** Sınıflandırıcı veya dedektör bir görüntüden emin olmadığında onu `veri/_incelenecek` klasörüne kaydeder (tespitlerde sadece o nesneyi kırpar, en sık kayıt ve en çok dosya ayarlanır).
- **Araçlar → Örnekleri Öğret (Ctrl+Shift+O):** Programın emin olmadığı görüntüleri tek tek gösterir; 1–9 tuşlarıyla doğru sınıfa koyun, Boşluk ile atlayın, Del ile silin, Ctrl+Z ile geri alın, yeni sınıf ekleyin. Hızlı Sınıflandırıcı yeni örnekleri görünce kendiliğinden yeniden eğitilir. Böylece program kullandıkça daha az yanılır.
- **Hızlı Sınıflandırıcı:** "Veri artırma" seçeneği (her örneğin aynalı, açık/koyu ve hafif döndürülmüş hâlleriyle de öğrenir) ve yeni **kesinlik** (certainty) çıkışı: görüntü başka bir sınıfa ne kadar yakınsa o kadar düşük. Eski "güven" çıkışı aynı kaldı.
- Hazır örnek: `examples/kendi_kendine_ogrenen.json`.

**Düzeltildi:**
- Hızlı Sınıflandırıcı, düz arka planlı ve birbirine çok benzeyen az sayıda örnekle (sınıf başına 3 resim gibi) eğitilince neredeyse rastgele cevap veriyordu (40 dairenin 9'unu bildi). Artık 40/40.
- Hızlı Sınıflandırıcı ve Örnek Topla, resmin dışına taşan bir kutu gelince resmin yanlış bir parçasını kesiyordu; bozuk (NaN) kutu bloğu durduruyordu.
- (Bu sürüm yayından önce bir hata avından geçti: toplanan her örnek sınıflandırıcıyı boşuna yeniden eğitiyordu, veri artırma güven puanını hep 1.0 yapıyordu, iki Öğret penceresi birbirinin silinenlerini kaybettiriyordu, klasör kutusunda Enter klasör seçiciyi açıyordu, aynı anda çalışan iki toplayıcı birbirinin dosyasının üzerine yazabiliyordu, "vida #12" gibi etiketler dosya adında bozuluyordu, var olan bir dosya adıyla yeni sınıf eklemek pencereyi çökertiyordu. Hepsi düzeltildi ve test edildi.)

**Testler:** 392 → 416. Tam takım üst üste 3 kez geçti.

## 1.2.1: Kendini Geliştir hata ayıklama
**Kritik:**
- 1.2.0'da program penceresi açıkken her `print` hata veriyordu. Bu yüzden bir blok başlatılamayınca akışın tamamı başlamıyordu. Ayrıca "Değer Göster" (günlüğe yaz açık) ve günlüğe yazan ~20 blok hata veriyordu.

**Güvenlik kuralları sıkılaştırıldı:**
- Korumalı bir dosyayla aynı adlı yeni bir klasör (ör. `ai/config/__init__.py`) korumalı dosyanın yerine geçebiliyordu. Artık engelleniyor. `visionflow/__init__.py` da korumalı, kuralları (evolve) değiştirmeye çalışan kod reddediliyor.
- Windows'ta klasör adının sonundaki nokta/boşluk (`ai./claude.py`) korumayı aşabiliyordu. Artık engelleniyor.
- Paketlenmiş programda önerilen eklenti, siz onaylamadan çalışan programa yükleniyordu. Artık ayrı, geçici bir süreçte deneniyor.

**Düzeltildi:**
- Eski bir değişikliği geri almak, aynı dosyadaki daha yeni bir değişikliği siliyordu. Artık dosya sonradan değiştiyse geri alma reddediliyor ve önce yenisini geri almanız isteniyor.
- Aynı dosyada birden fazla değişiklik yanlış yere uygulanabiliyordu. Artık sırayla kontrol ediliyor.
- Dosya yazılamazsa (Windows kilidi) değişiklik yarım kalıyordu. Artık her şey geri konuyor ve tekrar denenebiliyor.
- Test edildikten sonra program değiştiyse öneri eski test sonucuyla uygulanabiliyordu. Artık önce yeniden test ediliyor.
- Günlük inceleme aynı eski hata için her gün yeniden Claude'a soruyordu (boşa harcama). Artık sadece yeni hatalarda soruyor.
- Bekleyen bir öneri silinemiyordu; "Öneriyi sil" düğmesi eklendi, artık uygulanamayan öneriler kendiliğinden siliniyor.
- Sayı veya saat içeren hatalar günlüğü dolduruyor ve gerçek hataları siliyordu.
- Hata yeri: kullanıcı betikleri ve proje içindeki `.venv` program kodu sanılıyordu; başlatma hataları yer bilgisi almıyordu.
- Kendini Geliştir penceresi tek açılıyor; geri almadaki beklenmedik hatalar artık mesajla gösteriliyor.
- Testler pytest'i kendi içinde çalıştıran araçları (IDE) kapatabiliyordu.

**Testler:** 368 → 392. Tam takım üst üste 3 kez geçti; 14 yeni testin hepsi 1.2.0 kodunda başarısız oluyor.

## 1.2.0: Kendini Geliştir
**Yeni:**
- **Araçlar → Kendini Geliştir (Ctrl+Shift+G):** Program kullanılırken yaşadığı hataları kodun hangi satırında olduğuyla kaydeder, kendini inceler (hatalar, eklentiler, kendi testleri) ve Claude'dan kendi kodu için düzeltme veya istediğiniz bir iyileştirme ister.
- **Kendini sorgular:** Her öneri önce programın geçici bir kopyasında uygulanıp tüm testlerle denenir. Testler bozulursa hata Claude'a gösterilir ve öneriyi kendisi düzeltir (en çok 3 tur).
- **Onayla uygula, geri al:** Sadece testleri geçen öneri, farkı görüp onayladığınızda uygulanır. Eski dosyalar yedeklenir, Geçmiş sekmesinden tek tıkla geri alınır.
- **Her gün kendini incele** (varsayılan kapalı): Açılışta arka planda kendini inceler; hata varsa test edilmiş bir öneri hazırlayıp bekletir, asla kendisi uygulamaz.
- Korumalı dosyalar: harcama sınırı/ayarlar, Claude bağlantısı ve kendini geliştirme kuralları değiştirilemez. Paketlenmiş .exe'de sadece `plugins/` klasörüne eklenti yazılır.
- Arayüzsüz çalışmada da hatalar günlüğe yazılır. Komut satırından rapor: `python -m visionflow.evolve`.

**Düzeltildi:**
- Tam test takımı bütün testler geçse bile kapanırken çöküyor (çıkış kodu 139) ve başarısız görünüyordu; artık doğru sonuç dönüyor.

**Testler:** 322 → 368. Tam takım üst üste 3 kez geçti. Hatalı bir öneriyi yakalayıp ikinci turda düzelten gerçek bir deneme (kopyada tam test takımıyla) de yapıldı.

## 1.1.0: Önizlemeden bölge çizme, vardiya raporu, beşinci hata avı
**Yeni:**
- **Bölgeyi fareyle çizin:** Kırp, Renk Tanı, Çizgi Geçiş Sayacı, Bölgedeki Nesneler ve Perspektif Düzelt bloklarında koordinat yazmak yerine önizlemede çizin. Önizleme bu bloklarda bloğa giren görüntüyü gösterir ve mevcut bölgeyi sarıyla çizer. Ctrl+Z ile geri alınır.
- **Vardiya Raporu bloğu (toplam 120):** Vardiya sonlarında o vardiyanın sağlam/hatalı sayısını ve verimini metin olarak verir; Telegram veya e-postaya bağlayın. Sayaç sıfırlansa da doğru sayar, elle de rapor alınabilir.
- Görüntü üstüne yazılan Türkçe harfler (ş, ğ, ı, İ, ü, ö, ç) artık soru işareti olmuyor: Yazı Yaz, tespit etiketleri, grafik başlığı, sayaç ve bölge yazıları.

**15 hata düzeltildi (Türkçe harf sorunu yukarıda):**
- Güvenlik: Web Panel'de bir QR kodu, OCR veya mesaj metni, paneli izleyen herkesin tarayıcısında kod çalıştırabiliyordu.
- Güvenlik: "Yapay Zeka ile Akış Kur" mevcut akışı değiştirirken e-posta şifrelerini ve bot token'larını Claude'a gönderiyordu. Artık bunlar bilgisayardan çıkmıyor ve akışta korunuyor.
- Yapay zeka ile akış değiştirince blokların yeri, kapalı blokların durumu ve kimlikleri kayboluyordu.
- Etiketleme aracı, sadece data.yaml'ı olan (Roboflow/Ultralytics) veri setlerinde sınıf adlarını "nesne" ile değiştiriyor, segmentasyon çizgilerini siliyordu.
- Bozuk tek bir eklenti programın hiç açılmamasına yol açıyordu; artık o eklenti atlanıp günlükte yazılıyor.
- Daire Bul, Yuvarlaklık ayarı en yükseğe getirilince çöküyordu.
- Blok aramasında "istatistik", "izle" gibi yazınca İ ile başlayan bloklar bulunmuyordu; artık Türkçe harf farkı gözetilmiyor ("kirp" → Kırp).
- Büyük bir akışta "Tümünü göster"den sonra fare tekerleğiyle yakınlaştırma çalışmıyordu.
- YouTube oturumu süresi dolunca yükleme bir daha hiç çalışmıyordu; artık yeniden giriş istiyor.
- Tolerans Kontrolü, sürekli geçersiz (NaN) ölçümü her karede hatalı parça sayıyordu.
- CSV Kaydı, dosya Excel'de açıkken satırları kaybediyordu; artık bekletip dosya kapanınca yazıyor.
- Hafıza, içi boşaltılan veya metni olmayan (taranmış) bir belge yeniden eklenince eski metni hatırlamaya devam ediyordu.
- Eğitim penceresi, çıktının ortasından bölünen satırlarda "bitti" haberini kaçırabiliyordu.

**Testler:** 304 → 322. Her düzeltilen hata için test var (14 yeni test eski kodda başarısız oluyor). Tam takım üst üste 3 kez geçti.

## 1.0.0: Dördüncü hata avı, kalite kontrol blokları
**Daha önce taranmamış kısımlar (çalıştırıcı, editör penceresi, e-posta, Telegram, webhook, harcama sınırı) tarandı; 16 hata düzeltildi:**
- Güvenlik: Dışarıdan Tetik bloğu token yokken her isteği kabul ediyordu; tarayıcıda açık herhangi bir web sitesi akışı tetikleyebiliyordu. Artık token yoksa sadece bu bilgisayardan gelen JSON istekleri kabul ediliyor. Bozuk bir istek bloğu kilitleyemiyor.
- Güvenlik: E-posta gönderirken sunucu şifreli bağlantı sunmazsa şifre açık gönderiliyordu. Artık gönderilmiyor ve hata açıkça yazılıyor.
- Çalışırken bir kaynak bloğun herhangi bir ayarı değişince blok baştan başlıyordu: Klasördeki Resimler 1. resme dönüyor, e-posta/Telegram kuyruğu boşalıyordu. Artık sadece gerçekten gereken ayarlar yeniden başlatıyor.
- Akış durdurulunca kuyrukta bekleyen Telegram mesajları ve "okundu" işaretlenen e-postalar kayboluyordu. Artık bir mesaj ancak akışa verilince onaylanıyor; bekleyenler bir sonraki çalıştırmada geliyor ve hiçbir mesaja iki kez cevap verilmiyor.
- Kamera veya zamanlayıcı olmayan akışlarda (ör. Metin → Yapay Zekaya Sor → Dosyaya Kaydet) cevap hiç gelmiyordu. Artık akış, bekleyen iş bitene kadar çalışıyor.
- Kurulum testi veya ikinci bir pencereyi kapatmak, çökme sonrası kurtarma dosyasını siliyordu.
- Değişken Yaz/Oku blokları resim ile kullanılınca her karede hata veriyordu.
- E-posta kontrolü, internet bir an kesilince sonsuza kadar takılıp bir daha e-posta okumuyordu (artık 30 sn zaman aşımı var).
- Akış durmadan pencere kapatılınca program çökebiliyor, video dosyası bozuk kalıyordu.
- İnternette Araştır, uzun aramalarda raporun ilk kısmını ve kaynakları kaybediyordu.
- Kurtarılan akışlarda ve "Farklı kaydet" sonrasında göreli dosya yolları bulunamıyordu.
- Akış denetimi, e-posta hesabı bloğun kendisine girilmişken bile "hesap yok" hatası veriyordu; webhook ve haber akışı için sıralı işlem uyarısı yoktu.
- Arayüzsüz çalışmada tek kaynak (ör. video dosyası) açılamazsa program boşta sonsuza kadar dönüyordu; artık hata verip çıkıyor, servis yöneticisi yeniden başlatabiliyor.
- Video Kaydet, dosya açılamadığında (ör. Windows'ta Türkçe harfli klasör) hiçbir şey yazmadan "kaydediyor" görünüyordu.
- Günlük harcama sınırı, editör ve arayüzsüz çalıştırıcı aynı anda Claude kullanınca harcamanın bir kısmını saymıyordu.

**Yeni bloklar (4, toplam 119):**
- **Tolerans Kontrolü:** Ölçülen değer alt-üst sınır arasında mı bakar; sağlam/hatalı sayısını, verim yüzdesini ve nominalden sapmayı verir.
- **Tespiti Kırp:** Bulunan nesnelerden birini (en emin olunan, en büyük veya sıradaki) keser; kesilen parçayı Yazı Oku veya Görüntü Sınıflandır bloğuna verin.
- **Değeri Yumuşat:** Titreyen ölçümü son N değerin medyanı veya ortalamasıyla sakinleştirir.
- **Üretim Hızı:** Bir sayacın dakikada/saatte kaç arttığını ölçer; hız hedefin altına düşünce uyarır.

**Testler:** 284 → 304. Her düzeltilen hata için test var (14 yeni test eski kodda başarısız oluyor veya takılıyor). Tam takım üst üste 3 kez geçti.

## 0.9.0: Üçüncü hata avı, değişkenler, dışarıdan tetik, kurulum testi
**Kod yeniden tarandı; 18 hata düzeltildi:**
- Klasörü İzle, klasörde 5000'den çok dosya olunca en eski dosyaları "yeni" sanıp tekrar işliyordu. Artık sadece klasörde hâlâ duran dosyaları hatırlıyor, hiçbirini unutmuyor.
- Kamera veya IP kamera bağlantısı kopunca akış duruyordu. Artık birkaç saniyede bir yeniden bağlanıyor; video dosyası bitince ise eskisi gibi duruyor.
- YOLO: küçük giriş boyutlu modellerde çıktı biçimi yanlış tahmin ediliyordu; YOLOv10/YOLO26 gibi uçtan uca modeller (özellikle 1-2 sınıflı olanlar) yanlış okunuyordu; 384x640 gibi dikdörtgen modeller kare sanılıp bozuk sonuç veriyordu; sınıf adları liste olarak yazılmış modellerde isimler okunmuyordu.
- YOLO, Görüntü Sınıflandır, ArUco ve Hareket Algıla blokları çalışırken model veya ayar değişince yeniden başlatılana kadar eski ayarla çalışıyordu.
- Renk Tanı, resmin dışına taşan bölgede çöküyordu; artık bölge kırpılıyor, tamamen dışarıdaysa açık hata veriyor.
- Örnek akışlar ve kendi akışlarınız, program kısayoldan veya başka klasörden açılınca "veri", "models" gibi göreli klasörleri bulamıyordu. Artık akış dosyasının ve programın klasörüne de bakılıyor.
- Paketlenmiş programda (.exe) "YOLO Modeli Eğit" hiç çalışmıyordu. Artık bilgisayarda kurulu Python'u buluyor, yoksa ne yapılacağını söylüyor. Eğitim çıktısındaki Türkçe harfler de bozulmuyor.
- data.yaml, "no", "null" veya iki nokta içeren sınıf adlarında ve "#" içeren klasörlerde eğitimi bozuyordu.
- Video Üret, adında Türkçe harf geçen klasörlerdeki ("Masaüstü") resimleri Windows'ta göremiyordu.
- Excel'e Kaydet her satırda dosyayı baştan açıp kaydediyordu; hızlı akışta program yavaşlıyordu. Artık satırlar toplanıp saniyede bir yazılıyor, akış durunca kalanlar da yazılıyor. Formül gibi görünen metinler artık başına ' eklenmeden, metin olarak saklanıyor. Bozuk dosyada her karede tekrar denemiyor.
- Çizgi Geçiş Sayacı, uzun süre çalışınca geçmiş her nesneyi hafızada tutup büyüyordu.
- Dosyayı Taşı, dosya başka programda açıkken (ör. tarayıcı hâlâ yazıyorken) iki kopya bırakabiliyor ve bir daha denemiyordu. Artık yarım kopyayı siliyor ve 2 saniye sonra tekrar deniyor.
- AI Merkezi her Ctrl+M'de yeni bir pencere açıyor ve kapatılan pencereler bellekte kalıyordu; cevap beklerken "Yeni sohbet"e basılınca eski konuşma geri geliyordu.
- Belge Oku ve Hafıza, Not Defteri'nin "Unicode" (UTF-16) ve "BOM'lu UTF-8" kaydettiği metin dosyalarını bozuk okuyordu.

**Yeni bloklar (3, toplam 115):**
- **Değişken Yaz / Değişken Oku:** Bir değeri isimle saklar, aynı isimli bloklar her yerden okur (kablo çekmeden bağlantı). "Ekle" modu sayaç gibi çalışır, "Kalıcı" seçilirse program kapanınca unutulmaz. Akış denetimi, karşılığı olmayan değişken adlarını uyarır.
- **Dışarıdan Tetik (Webhook Al):** Program içinde küçük bir web adresi açar. Form, Zapier/Make, telefon veya başka bir program istek atınca akış çalışır; JSON, form ve adres parametreleri okunur. Varsayılan olarak sadece bu bilgisayardan erişilir; yerel ağa açmak için token zorunlu.

**Kurulum testi:** `VisionFlow --selftest` blokları, örnekleri, OpenCV'yi, isteğe bağlı paketleri, ffmpeg'i ve pencereyi kontrol edip rapor yazar. `VisionFlow --headless akis.json` paketlenmiş programda arayüzsüz çalıştırır. Linux paketi bu iki komutla denendi.

**Örnekler:** "sipariş toplayıcı" (yeni): webhook → bilgi çıkar → Excel + günlük sayaç + Telegram.

**Testler:** 252 → 284. Her düzeltilen hata için test var (19 yeni test eski kodda başarısız oluyor). Tam takım üst üste 3 kez geçti.

## 0.8.0: İkinci kapsamlı hata ayıklama, 4 yeni blok
**Editör, bloklar ve dosya işlemleri baştan tarandı; 16 hata düzeltildi:**
- Düzen menüsünden veya Ctrl+V ile yapıştırma hiç çalışmıyordu (sadece sağ tık menüsü çalışıyordu).
- Ondalıklı ayarlarda yazılan değer kaydırıcı yüzünden değişiyordu (0,29 → 0,28).
- Çizgi Geçiş Sayacı, nesnenin merkezi tam çizginin üstüne denk gelince geçişi saymıyordu.
- Etiketleme aracı, resmin dışına taşan kutuları geçersiz YOLO etiketi olarak kaydediyordu.
- Örnek Topla, dedektör bir şey bulamayınca bütün kareyi kaydedip eğitim verisini bozuyordu.
- Seri Port Gönder, Türkçe harfleri ve ters bölü işaretini bozuyordu.
- Kaydetme başarısız olunca (klasör yok, yetki yok) program hata verip pencerenin dosya yolunu değiştiriyordu; kaydetme yarıda kalırsa eski dosya bozulabiliyordu. Artık kayıt güvenli: önce geçici dosyaya yazılır.
- Elle düzenlenmiş veya bozuk akış dosyaları açılırken çöküyor, yanlış tipteki ayarlar (ör. "640" yazı olarak) blokları bozuyordu. Artık değerler düzeltiliyor, düzeltilemeyenler uyarıyla varsayılana dönüyor.
- Listede olmayan bir seçenek değeri yanlış gösteriliyordu.
- Yapay zekanın bozuk cevapları akış kurucuyu çökertiyordu; artık düzeltme turuna giriyor.
- Veritabanı Kaydı, aynı veya ayrılmış sütun adlarında ("x,x,y", "id") ve sütun adları değişince çalışmıyordu.
- Video Kaydet, aynı saniyede başlayan iki kaydın ilkini siliyordu.
- Çalışırken değiştirilen bazı ayarlar (veritabanı tablosu, web panel portu, PLC adresi, metin koşulu…) yeniden başlatana kadar etkisizdi; artık blok kendini yeniden başlatıyor.
- Arayüzsüz çalışma servis/Docker tarafından durdurulunca video kayıtları bozuk kalıyordu.
- İstatistik bloğu ve Web Panel, tek bir "NaN" değerinde bozuluyordu.
- Başka yerden kopyalanmış bozuk bir akış yapıştırılınca yarım kalıyordu.
- Ayrıca: kısa Modbus cevabı artık açık hata veriyor, "examples" ile başlayan klasörlerdeki akışlar örnek sanılmıyordu.

**Yeni bloklar (4, toplam 112):** Haber Akışı (RSS/Atom, kelime filtresi, sadece yeni haberler), Gecikme, Günlük Sayaç (vardiya saatinde sıfırlanır, program kapanınca unutmaz), Excel'e Kaydet (gerçek .xlsx; dosya Excel'de açıkken satırlar kaybolmaz).

**Örnekler:** "haber ozeti telegram" (yeni).

**Testler:** 218 → 252. Her düzeltilen hata için ayrı test var (24 test eski kodda başarısız oluyor). Tam takım üst üste 5 kez geçti.

## 0.7.0: Kapsamlı hata ayıklama, akış denetimi, 5 yeni blok
**Kodun tamamı hata için tarandı; bulunan 13 hata ve birkaç küçük sorun düzeltildi:**
- Bir blok önizlemeye alışılmadık bir resim verince (ör. Python Kodu ile `img > 128` maskesi, 4 kanallı resim) akış sessizce duruyordu. Artık her tür resim gösteriliyor.
- Belge Oku, yeni dosya okunamayınca önceki dosyanın yazısını vermeye devam ediyordu.
- 20'den fazla okunmamış e-posta varsa eskiler hiç işlenmiyordu. "noreply" e-postaları da sırayı tıkayabiliyordu.
- Program yeniden açıldığında son işlenen e-postalar unutulabiliyor, aynı e-postaya tekrar cevap yazılabiliyordu.
- Hafızaya klasör eklerken farklı klasörlerdeki aynı adlı dosyalar (2023/fatura.txt, 2024/fatura.txt) birbirinin üstüne yazılıyordu.
- Günlük harcama sınırı, dosya yazılırken okunursa 0 $ görünüp aşılabiliyordu.
- Onay klasörü modunda aynı kişiye aynı saniyede yazılan iki cevaptan biri kayboluyordu.
- E-postayı Cevapla bloğunun sayacı arka arkaya cevaplarda hep 1 gösteriyordu.
- Yapay zeka bloklarında nadiren sıraya alınan bir istek hiç çalışmıyordu. Bir hatadan sonra gelen başarılı cevap hatayı silmiyordu. Sayı yerine "NaN" gelince her karede ücretli istek atılıyordu.
- Aynı bağlantı iki kez çekilince ekranda hayalet bir çizgi kalıyordu.
- Kaynak bloklar ("sonraki" bağlı değilken) art arda öğe verince sonraki blok sadece ilkini görüyordu.
- Web Sayfası Oku, bilinmeyen karakter kodlaması bildiren sayfalarda hata veriyordu.
- Saatte Çalıştır, tam o dakika içinde başlatılınca o saatin işini atlıyordu.
- Boş (0 piksel) resimler bloklarda anlaşılmaz hatalar veriyordu; artık açık bir mesaj var. Küçük resmi küçültmek ve resmin dışını kırpmak düzgün hata veriyor.

**Yeni özellik: Akışı denetle** (`Ctrl+Shift+D`, `headless --check`): bağlanmamış bloklar, eksik anahtar/hesap/token/adres ve sıralı işlem bağlantısı eksikleri. Çalıştırırken bulunan hatalar kayıt paneline de yazılır.

**Yeni bloklar (5, toplam 108):** Metin Koşulu, Saat Aralığında, API'den Veri Al, Discord / Slack / Teams Gönder, Dosyayı Taşı / Kopyala.

**Örnekler:** "doviz kuru bildirimi" (yeni); fatura okuyucu işlenen dosyayı taşıyor.

**Testler:** 145 → 218. Her düzeltilen hata için ayrı test var; bu testlerin 23'ü eski kodda başarısız oluyor. Tam takım üst üste 5 kez geçti.

## 0.6.0: Daha fazla test, yeni otomasyon blokları, hata düzeltmeleri
**Yeni bloklar (6, toplam 103):**
- **Telegram Mesaj Al:** Botunuza gelen mesajları sırayla verir. Sadece izinli chat ID'lerden gelenler işlenir; izinsizlerin ID'si kayda yazılır.
- **Web Sayfası Oku:** Bir sayfanın yazısını belirli aralıklarla okur, değişince haber verir (fiyat, duyuru takibi).
- **Klasörü İzle:** Klasöre yeni dosya gelince verir; dosya tamamen kopyalanana kadar bekler.
- **Belge Oku:** PDF, Word, e-posta, metin dosyasının yazısını çıkarır.
- **Saatte Çalıştır:** "09:00, 17:30" gibi saatlerde tetikler; her gün, hafta içi veya hafta sonu.
- **Bilgi Çıkar:** Claude ile metinden istediğiniz alanları (firma, tutar, tarih…) düzenli olarak çıkarır.

**Yeni özellikler:**
- **Günlük harcama sınırı** (varsayılan 5 $) ve AI Merkezi'nde bugünkü harcama.
- **Sıralı işleme:** Kaynak blokların "sonraki" girişine son adımın "done" çıkışı bağlanabilir (geri bağlantı artık döngü hatası vermiyor). Son adım takılırsa 10 dakika sonra sıradakine geçilir.
- Telegram Gönder'e **chat ID girişi**: botun cevabı, mesajı yazan kişiye gider. Token boşsa AI Merkezi ayarı kullanılır.
- Klasör ayarlarında klasör seçme penceresi; kod ve uzun metin ayarları geniş kutuda.

**Örnekler:** "telegram asistani", "fatura okuyucu"; "eposta asistani" artık e-postaları sırayla işliyor.

**Düzeltilen hatalar:**
- Yapay zeka cevap yazarken ikinci bir e-posta gelirse, cevap yanlış kişiye gidebiliyordu.
- Yapay zeka blokları meşgulken gelen istekler kayboluyordu; artık sıraya alınıyor.
- E-postalar sunucu numarasına (UID) göre hatırlandığı için klasör değişince aynı e-posta tekrar cevaplanabiliyordu; artık Message-ID kullanılıyor. "noreply" adreslerine cevap yazılmıyor.
- Uzun Telegram mesajları ve fotoğraf açıklamaları reddediliyordu (4096 / 1024 karakter sınırı).
- Arayüzsüz çalıştırma kamerasız akışlarda işlemciyi %100 kullanıyordu; varsayılan 30 kare/sn.
- Yazıyla kurulan akışlarda bloklar üst üste biniyordu (gerçek blok yükseklikleri kullanılıyor).

**Testler:** 123 → 145 (sahte Telegram sunucusu, sıralı e-posta, harcama sınırı, klasör izleme, zamanlama, fatura örneği uçtan uca). Tam test takımı üst üste 3 kez geçti.

## 0.5.0: AI Merkezi
**Yapay zeka (Claude):**
- **AI Merkezi penceresi** (`Ctrl+M`): kendi yapay zekanızla sohbet, hafıza yönetimi, hesap ayarları.
- **Kendi hafızanız:** PDF, Word, metin, e-posta belgelerinden öğrenen yerel bilgi bankası (bilgisayarınızda kalır); sorulara kendi bilgilerinizle cevap verilir. Önceki konuşmaları hatırlama.
- **Yazıyla akış kurma** (`Ctrl+K`): ne istediğinizi yazın, Claude blokları kurup bağlasın. Hatalı öneriler kontrol edilip Claude'a düzelttirilir.

**Yeni bloklar (12, toplam 97):** Yapay Zekaya Sor, Yapay Zeka ile Sınıflandır, İnternette Araştır, Hafızaya Ekle, Hafızada Ara, E-posta Oku, E-postayı Cevapla (varsayılan: taslağa kaydeder), Seslendir (ücretsiz doğal Türkçe ses, internetsiz ses veya ElevenLabs), Video Üret (resim + ses + alt yazı → MP4, yatay/dikey), YouTube'a Yükle (varsayılan: gizli), Metin, Metni Dosyaya Kaydet.

**Diğer:** Zamanlayıcı artık kamerasız akışları da sürekli çalıştırıyor ve "başlarken hemen tetikle" seçeneği var.

**Örnekler:** "eposta asistani", "arastir seslendir video".

**Testler:** 89 → 123 (sahte Claude, sahte IMAP sunucusu, gerçek ffmpeg ile video, çevrimdışı ses motoru, uçtan uca e-posta ve video akışları).

## 0.4.0
**Yeni bloklar (12, toplam 85):**
- **Klasördeki Resimler:** Bir klasördeki resimleri sırayla verir (toplu kontrol, model deneme). Resim ekranda beklerken ayarlar canlı değiştirilebilir.
- **Şekil Tanı:** Konturları üçgen, kare, dikdörtgen, beşgen, altıgen, daire, elips olarak sınıflandırır.
- **Daire Bul (Hough):** Delik, vida, kapak, tablet sayar; çapı piksel olarak yazar.
- **Bölgedeki Nesneler:** Çizdiğiniz çokgen alanın içindeki tespitleri sayar ("tehlikeli bölgede kişi var" alarmı).
- **Sınıfa Göre Say:** YOLO tespitlerini sınıf adına göre ayrı ayrı sayar.
- **Perspektif Düzelt**, **Keskinleştir / Gürültü Azalt**, **Arka Plan Çıkar** (sabit kamerada yeni gelen nesne).
- **E-posta Gönder** (Gmail dahil, resim ekli), **Telegram Gönder** (fotoğraflı), bekleme süresiyle spam yapmaz.
- **Veritabanı Kaydı (SQLite):** Kurulum gerektirmeyen tek dosyalık veritabanı; Excel / Power BI ile açılır.
- **Grafik:** Bir veya iki değerin zamana göre çizgi grafiği.

**Editör:**
- Otomatik kayıt ve çökme sonrası kurtarma (30 sn'de bir yedek).
- **Görünüm → Otomatik yerleştir** (`Ctrl+Shift+L`), geri alınabilir.
- **Yardım → Kısayollar** (`F1`) ve Hakkında penceresi.
- Pencere boyutu ve panel yerleşimi hatırlanıyor.

**Örnekler:** "klasorden sekil tanima", "bolge alarmi veritabani".

**Düzeltilen hatalar:**
- Bir blok başlatılamayınca (kamera takılı değil, model dosyası yok) bütün akış duruyordu; artık sadece o blok hata gösterir.
- Bilinmeyen blok içeren (ör. eklentisi eksik) akış dosyası hiç açılmıyordu; artık o bloklar atlanıp uyarı gösteriliyor.
- Kopyala/yapıştırda devre dışı bloklar etkin hale geliyordu.
- Model yolu gibi bir metin ayarına tıklayıp değiştirmeden çıkınca model yeniden yükleniyordu.
- Web Panel durdurulunca canlı yayın bağlantıları açık kalıyordu.
- Arayüzsüz çalıştırmada aynı hata her karede tekrar yazılıyordu.
- Kaynağı (kamera) başlatılamayan akış işlemciyi boşuna meşgul ediyordu.

**Testler:** 52 → 89 (yerel SMTP sunucusuyla e-posta, yerel HTTP ile Telegram, SQLite, kurtarma dosyası, otomatik yerleşim).

## 0.3.0
**Kendi modelini eğit:**
- **Hızlı Sınıflandırıcı:** Klasördeki örneklerden saniyeler içinde öğrenir (GPU, internet gerekmez); klasöre resim eklenince kendini yeniden eğitir. Örnek akışta 600 tespitin 600'ünü doğru sınıflandırdı.
- **Anomali Tespiti:** Sadece sağlam ürün resimleriyle öğrenir, çizik/leke/eksik parçayı ısı haritasıyla gösterir.
- **Örnek Topla** ve **YOLO Veri Seti Kaydet** (mevcut modelle otomatik etiketleme) blokları.
- **Araçlar → Etiketleme Aracı:** Kutu çiz, sınıf seç (1–9), A/D ile gez; YOLO formatında kaydeder, data.yaml oluşturur.
- **Araçlar → YOLO Modeli Eğit:** Eğitimi ayrı süreçte çalıştırır, bitince ONNX'i models/ klasörüne koyar (ultralytics gerekir).

**Yeni bloklar (12, toplam 73):** Yazı Oku (OCR, Türkçe dahil), YOLO Segmentasyon, Siemens S7 Oku/Yaz, OPC UA Oku/Yaz, Web Panel (aynı ağdaki telefon/bilgisayardan canlı görüntü ve değerler), Not, ve yukarıdaki öğrenme blokları.

**Editör:** Tab ile hızlı blok ekleme, blok devre dışı bırakma (Ctrl+E, giriş doğrudan çıkışa geçer), tuvalde not kutuları, kaydırılabilir ayar paneli, uzun blok adları kısaltılarak gösteriliyor.

**Dağıtım:** PyInstaller tarifi ve `paket_olustur.bat` / `.sh` ile Python kurulu olmayan bilgisayarlar için tek klasörlük paket.

**Düzeltilen hatalar:**
- Bir bloktan türetilen yeni blok, ana bloğun yerine kaydoluyordu (YOLO Segmentasyon eklenince YOLO Tespit kayboluyordu).
- Hızlı Sınıflandırıcı, klasöre yazılmakta olan bir resim yüzünden yeniden eğitilemezse akış duruyordu; artık eski modelle devam ediyor.

**Testler:** 32 → 52 (sahte S7 sunucusu, gerçek OPC UA sunucusu, Tesseract OCR, web panel HTTP, etiketleme aracı).


## 0.2.0
**Yeni bloklar (26, toplam 61):** QR/Barkod Oku, ArUco, Hareket Algıla, Referansla Karşılaştır, Netlik Ölç, Renk Tanı, Nesne Ölç (mm), Parlaklık/Kontrast, CLAHE, Yan Yana Birleştir, Yazı Yaz, Bölge/Çizgi Çiz, Nesne Takibi, Çizgi Geçiş Sayacı, Görüntü Sınıflandır (ONNX), Zamanlayıcı, Sinyali Uzat, Mandal, Seçici, Metin Oluştur, İstatistik, PLC Yaz/Oku (Modbus TCP), MQTT Gönder, Seri Port Gönder, Video Kaydet.

**Editör:** Geri al / yinele, kopyala / yapıştır (bağlantılarıyla), sağ tık menüsü, tümünü seç, son açılan dosyalar, kaydedilmemiş değişiklik uyarısı (başlıkta *), araç çubuğunda Maks FPS ayarı, durum çubuğunda hata listesi.

**Düzeltilen hatalar:**
- Ayar kutusuna yazarken `Del` veya `F` tuşu bloğu siliyor / görünümü değiştiriyordu.
- Kırmızı renk (ton 170 → 10 gibi 180'i aşan aralık) filtrelenemiyordu.
- Döngü oluşturan bir bağlantı reddedilince o girişteki eski bağlantı da kayboluyordu.
- OpenCV 5 kurulunca Yüz Tespiti çöküyordu (sürüm sabitlendi, anlaşılır hata mesajı eklendi).
- Test Görüntüsü bloğu bekleme süresini kendi işlem süresi gibi gösteriyordu (yerine genel FPS sınırlayıcı geldi).
- "Hareketli" ayarı çalışırken açılınca akış durağan kalıyordu.
- Webhook gönderilemediğinde konsola hata yığını basıyordu; artık günlüğe tek satır yazıyor.
- Kontur Çiz ve örnek eklenti, giriş bağlı değilken anlaşılmaz hata veriyordu.
- Pencere kapanırken arka planda hata oluşuyordu.
- Windows'ta USB kameralar daha hızlı açılsın diye DirectShow kullanılıyor.

**Testler:** 12 → 32 test (blok davranışları, sahte PLC ile Modbus, arayüz: geri al, kopyala, kısayollar, canlı ayar, çalışırken yeniden bağlama).
