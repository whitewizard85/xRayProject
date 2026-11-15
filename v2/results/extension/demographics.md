# Caratteristiche demografiche

| Dataset | Unita' | n | Eta' nota | Eta' media ± SD | Mediana [IQR] | <18 anni | Sesso noto | Donne (% dei noti) |
|---|---|---:|---:|---|---|---:|---:|---|
| NIH train | image | 79269 | 79257 | 46.9 ± 16.4 | 49 [35–59] | 3457 | 79269 | 34685 (43.8%) |
| NIH train | patient (first image) | 21563 | 21557 | 46.1 ± 16.7 | 48 [34–58] | 1048 | 21563 | 9935 (46.1%) |
| NIH val | image | 16565 | 16562 | 47.5 ± 17.1 | 50 [35–60] | 819 | 16565 | 7181 (43.4%) |
| NIH val | patient (first image) | 4621 | 4619 | 46.4 ± 16.8 | 48 [34–58] | 235 | 4621 | 2126 (46.0%) |
| NIH test | image | 16286 | 16285 | 45.9 ± 16.8 | 48 [33–59] | 965 | 16286 | 6914 (42.5%) |
| NIH test | patient (first image) | 4621 | 4621 | 45.8 ± 16.7 | 47 [34–58] | 246 | 4621 | 2114 (45.7%) |
| NIH test, images with bounding boxes | image | 131 | 131 | 47.9 ± 17.0 | 52 [34–59] | 8 | 131 | 53 (40.5%) |
| VinDr-CXR external test (15 000) | image | 15000 | 3760 | 54.4 ± 17.6 | 57 [42–67] | 107 | 7354 | 3514 (47.8%) |
| VinDr-CXR, images with bounding boxes | image | 3602 | 2131 | 60.9 ± 14.6 | 62 [54–70] | 21 | 3187 | 1586 (49.8%) |

Note:
- NIH: 16 immagini con eta' > 100 anni nel file originale, trattate come mancanti.
- VinDr: intestazioni non leggibili: 0; eta' mancante o non interpretabile: 11240; sesso mancante o diverso da M/F: 7646.
- VinDr: l'identificativo paziente non e' disponibile nella release, quindi i conteggi sono per immagine.
