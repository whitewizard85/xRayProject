# Estensione 2026-11: report automatico

_Numeri da verificare sui CSV prima di riportarli nel paper._

- ConvNeXt-Base (paper recipe): 10 run valutati
- DenseNet-121 (paper recipe): 10 run valutati
- ConvNeXt-Base, wBCE/focal (paper recipe): 10 run valutati
- ConvNeXt-Base (alternative regime): 10 run valutati
- Swin-B (paper recipe): 10 run valutati

## Discriminazione (classi comuni) e variabilita' tra seed

- ConvNeXt-Base (paper recipe): AUC interna 0.838, esterna 0.901; SD tra seed 0.0015 vs 0.0112, rapporto 7.6 [IC 2.9, 19.7], leave-one-seed-out 5.8–9.6
- DenseNet-121 (paper recipe): AUC interna 0.827, esterna 0.869; SD tra seed 0.0012 vs 0.0054, rapporto 4.6 [IC 1.3, 16.5], leave-one-seed-out 3.6–6.9
- ConvNeXt-Base, wBCE/focal (paper recipe): AUC interna 0.834, esterna 0.894; SD tra seed 0.0020 vs 0.0058, rapporto 2.9 [IC 0.4, 21.9], leave-one-seed-out 1.6–5.2
- ConvNeXt-Base (alternative regime): AUC interna 0.834, esterna 0.902; SD tra seed 0.0019 vs 0.0055, rapporto 2.8 [IC 0.7, 11.8], leave-one-seed-out 2.1–4.0
- Swin-B (paper recipe): AUC interna 0.836, esterna 0.904; SD tra seed 0.0015 vs 0.0049, rapporto 3.3 [IC 2.1, 5.1], leave-one-seed-out 3.0–3.6

## Epoca selezionata (miglior AUC di validazione)

- ConvNeXt-Base (paper recipe) / asl: media 2.0 (min 2, max 2) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.057
- ConvNeXt-Base (paper recipe) / bce: media 3.0 (min 3, max 3) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.054
- DenseNet-121 (paper recipe) / asl: media 3.4 (min 3, max 4) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.015
- DenseNet-121 (paper recipe) / bce: media 4.6 (min 4, max 5) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.016
- ConvNeXt-Base, wBCE/focal (paper recipe) / focal: media 2.4 (min 2, max 3) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.049
- ConvNeXt-Base, wBCE/focal (paper recipe) / wbce: media 3.4 (min 3, max 4) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.026
- ConvNeXt-Base (alternative regime) / asl: media 5.0 (min 5, max 5) su 15 epoche; calo AUC val dal picco all'ultima epoca 0.029
- ConvNeXt-Base (alternative regime) / bce: media 5.8 (min 5, max 7) su 15 epoche; calo AUC val dal picco all'ultima epoca 0.013
- Swin-B (paper recipe) / asl: media 3.2 (min 3, max 4) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.066
- Swin-B (paper recipe) / bce: media 3.4 (min 3, max 4) su 10 epoche; calo AUC val dal picco all'ultima epoca 0.046

## Calibrazione (ECE con 10 bin pesati; media sui seed)

- ConvNeXt-Base (paper recipe) / asl: interna raw 0.294, TS 0.078, Platt 0.004 (intercetta raw -1.76); esterna raw 0.318, TS 0.105, Platt 0.047
- ConvNeXt-Base (paper recipe) / bce: interna raw 0.012, TS 0.006, Platt 0.004 (intercetta raw -0.19); esterna raw 0.041, TS 0.047, Platt 0.048
- DenseNet-121 (paper recipe) / asl: interna raw 0.288, TS 0.085, Platt 0.004 (intercetta raw -1.89); esterna raw 0.314, TS 0.114, Platt 0.049
- DenseNet-121 (paper recipe) / bce: interna raw 0.008, TS 0.006, Platt 0.004 (intercetta raw -0.29); esterna raw 0.041, TS 0.045, Platt 0.046
- ConvNeXt-Base, wBCE/focal (paper recipe) / focal: interna raw 0.133, TS 0.005, Platt 0.004 (intercetta raw -0.02); esterna raw 0.161, TS 0.047, Platt 0.047
- ConvNeXt-Base, wBCE/focal (paper recipe) / wbce: interna raw 0.225, TS 0.228, Platt 0.004 (intercetta raw -3.13); esterna raw 0.238, TS 0.246, Platt 0.047
- ConvNeXt-Base (alternative regime) / asl: interna raw 0.284, TS 0.086, Platt 0.004 (intercetta raw -1.88); esterna raw 0.312, TS 0.114, Platt 0.049
- ConvNeXt-Base (alternative regime) / bce: interna raw 0.010, TS 0.007, Platt 0.004 (intercetta raw -0.22); esterna raw 0.046, TS 0.046, Platt 0.047
- Swin-B (paper recipe) / asl: interna raw 0.292, TS 0.081, Platt 0.004 (intercetta raw -1.89); esterna raw 0.322, TS 0.113, Platt 0.050
- Swin-B (paper recipe) / bce: interna raw 0.010, TS 0.006, Platt 0.004 (intercetta raw -0.27); esterna raw 0.044, TS 0.048, Platt 0.049

## Localizzazione VinDr: effetto ASL - BCE (appaiato per seed, IC t al 95%)

- ConvNeXt-Base (paper recipe) / gradcam / energy: +0.0377 [+0.0084, +0.0670] (BCE 0.328, ASL 0.365)
- ConvNeXt-Base (paper recipe) / gradcam / norm_energy: +0.4964 [+0.0734, +0.9194] (BCE 6.884, ASL 7.380)
- ConvNeXt-Base (paper recipe) / gradcampp / energy: -0.0152 [-0.0271, -0.0032] (BCE 0.347, ASL 0.332)
- ConvNeXt-Base (paper recipe) / gradcampp / norm_energy: -0.6897 [-1.1507, -0.2287] (BCE 7.599, ASL 6.909)
- ConvNeXt-Base (paper recipe) / ig / energy: +0.0120 [+0.0021, +0.0219] (BCE 0.172, ASL 0.184)
- ConvNeXt-Base (paper recipe) / ig / norm_energy: +0.2986 [-0.0495, +0.6467] (BCE 3.843, ASL 4.141)
- ConvNeXt-Base (paper recipe) / ig_blur / energy: +0.0034 [-0.0029, +0.0096] (BCE 0.157, ASL 0.160)
- ConvNeXt-Base (paper recipe) / ig_blur / norm_energy: +0.0788 [-0.3239, +0.4815] (BCE 4.701, ASL 4.780)
- DenseNet-121 (paper recipe) / gradcam / energy: -0.0730 [-0.0922, -0.0538] (BCE 0.316, ASL 0.243)
- DenseNet-121 (paper recipe) / gradcam / norm_energy: -1.1395 [-1.5285, -0.7505] (BCE 5.632, ASL 4.493)
- DenseNet-121 (paper recipe) / gradcampp / energy: -0.0332 [-0.0396, -0.0268] (BCE 0.142, ASL 0.109)
- DenseNet-121 (paper recipe) / gradcampp / norm_energy: -0.5686 [-0.6312, -0.5060] (BCE 2.430, ASL 1.861)
- DenseNet-121 (paper recipe) / ig / energy: -0.0088 [-0.0185, +0.0010] (BCE 0.148, ASL 0.140)
- DenseNet-121 (paper recipe) / ig / norm_energy: -0.2162 [-0.4376, +0.0052] (BCE 2.878, ASL 2.661)
- DenseNet-121 (paper recipe) / ig_blur / energy: -0.0092 [-0.0173, -0.0012] (BCE 0.123, ASL 0.114)
- DenseNet-121 (paper recipe) / ig_blur / norm_energy: -0.2977 [-0.6103, +0.0149] (BCE 3.138, ASL 2.841)
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcam / energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcam / norm_energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcampp / energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcampp / norm_energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig / energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig / norm_energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig_blur / energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig_blur / norm_energy: +nan [+nan, +nan] (BCE nan, ASL nan)
- ConvNeXt-Base (alternative regime) / gradcam / energy: +0.0010 [-0.0139, +0.0158] (BCE 0.305, ASL 0.306)
- ConvNeXt-Base (alternative regime) / gradcam / norm_energy: +0.1174 [-0.3904, +0.6252] (BCE 5.599, ASL 5.716)
- ConvNeXt-Base (alternative regime) / gradcampp / energy: -0.0251 [-0.0419, -0.0082] (BCE 0.332, ASL 0.306)
- ConvNeXt-Base (alternative regime) / gradcampp / norm_energy: -0.3294 [-1.0260, +0.3672] (BCE 6.420, ASL 6.090)
- ConvNeXt-Base (alternative regime) / ig / energy: -0.0075 [-0.0128, -0.0023] (BCE 0.163, ASL 0.155)
- ConvNeXt-Base (alternative regime) / ig / norm_energy: -0.2758 [-0.4730, -0.0786] (BCE 3.354, ASL 3.078)
- ConvNeXt-Base (alternative regime) / ig_blur / energy: -0.0104 [-0.0136, -0.0071] (BCE 0.149, ASL 0.139)
- ConvNeXt-Base (alternative regime) / ig_blur / norm_energy: -0.4354 [-0.5536, -0.3173] (BCE 4.072, ASL 3.637)
- Swin-B (paper recipe) / gradcam / energy: +0.0208 [-0.0191, +0.0606] (BCE 0.261, ASL 0.282)
- Swin-B (paper recipe) / gradcam / norm_energy: +0.2296 [-0.5925, +1.0518] (BCE 4.340, ASL 4.570)
- Swin-B (paper recipe) / gradcampp / energy: -0.1134 [-0.2010, -0.0259] (BCE 0.279, ASL 0.165)
- Swin-B (paper recipe) / gradcampp / norm_energy: -1.8935 [-3.6326, -0.1543] (BCE 4.890, ASL 2.996)
- Swin-B (paper recipe) / ig / energy: -0.0018 [-0.0076, +0.0040] (BCE 0.184, ASL 0.182)
- Swin-B (paper recipe) / ig / norm_energy: -0.0929 [-0.4410, +0.2552] (BCE 4.348, ASL 4.255)
- Swin-B (paper recipe) / ig_blur / energy: -0.0060 [-0.0130, +0.0010] (BCE 0.162, ASL 0.156)
- Swin-B (paper recipe) / ig_blur / norm_energy: -0.2322 [-0.6363, +0.1719] (BCE 4.891, ASL 4.659)

## Accordo fra mappe di seed diversi (stessa condizione, VinDr)

- ConvNeXt-Base (paper recipe) / gradcam: rho medio 0.699
- ConvNeXt-Base (paper recipe) / gradcampp: rho medio 0.633
- ConvNeXt-Base (paper recipe) / ig: rho medio 0.874
- ConvNeXt-Base (paper recipe) / ig_blur: rho medio 0.915
- DenseNet-121 (paper recipe) / gradcam: rho medio 0.608
- DenseNet-121 (paper recipe) / gradcampp: rho medio 0.786
- DenseNet-121 (paper recipe) / ig: rho medio 0.861
- DenseNet-121 (paper recipe) / ig_blur: rho medio 0.896
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcam: rho medio 0.621
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcampp: rho medio 0.604
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig: rho medio 0.847
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig_blur: rho medio 0.902
- ConvNeXt-Base (alternative regime) / gradcam: rho medio 0.706
- ConvNeXt-Base (alternative regime) / gradcampp: rho medio 0.667
- ConvNeXt-Base (alternative regime) / ig: rho medio 0.891
- ConvNeXt-Base (alternative regime) / ig_blur: rho medio 0.927
- Swin-B (paper recipe) / gradcam: rho medio 0.597
- Swin-B (paper recipe) / gradcampp: rho medio 0.349
- Swin-B (paper recipe) / ig: rho medio 0.851
- Swin-B (paper recipe) / ig_blur: rho medio 0.916

## Randomizzazione a cascata: rho originale vs randomizzato dopo l'ultimo livello (VinDr)

- ConvNeXt-Base (paper recipe) / gradcam: -0.008 ± 0.184
- ConvNeXt-Base (paper recipe) / gradcampp: -0.010 ± 0.203
- ConvNeXt-Base (paper recipe) / ig_black: 0.539 ± 0.126
- ConvNeXt-Base (paper recipe) / ig_blur: 0.646 ± 0.096
- DenseNet-121 (paper recipe) / gradcam: -0.027 ± 0.206
- DenseNet-121 (paper recipe) / gradcampp: -0.057 ± 0.207
- DenseNet-121 (paper recipe) / ig_black: 0.723 ± 0.100
- DenseNet-121 (paper recipe) / ig_blur: 0.740 ± 0.082
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcam: -0.000 ± 0.181
- ConvNeXt-Base, wBCE/focal (paper recipe) / gradcampp: -0.013 ± 0.190
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig_black: 0.536 ± 0.131
- ConvNeXt-Base, wBCE/focal (paper recipe) / ig_blur: 0.650 ± 0.096
- ConvNeXt-Base (alternative regime) / gradcam: -0.008 ± 0.171
- ConvNeXt-Base (alternative regime) / gradcampp: -0.008 ± 0.197
- ConvNeXt-Base (alternative regime) / ig_black: 0.549 ± 0.124
- ConvNeXt-Base (alternative regime) / ig_blur: 0.623 ± 0.089
- Swin-B (paper recipe) / gradcam: 0.057 ± 0.191
- Swin-B (paper recipe) / gradcampp: 0.019 ± 0.177
- Swin-B (paper recipe) / ig_black: 0.569 ± 0.139
- Swin-B (paper recipe) / ig_blur: 0.667 ± 0.089

## IG con baseline nera vs sfocata (VinDr, media sui run)

- loc_*_energy: nera 0.167, sfocata 0.146
- loc_*_norm_energy: nera 3.619, sfocata 4.149
- faith_*: nera 0.010, sfocata 0.039
- cascade_full_*: nera 0.583, sfocata 0.665

## Ipotesi pre-specificate (famiglia H1-H5, correzione di Holm)

- **H1 swin** rapporto SD esterna/interna > 1: stima 3.311 [2.15, 5.10]; p = 0.000771; p Holm = 0.00231
- **H2 swin** ASL: ECE interna TS - Platt > 0: stima 0.07629 [0.0673, 0.0853]; p = 9.53e-06; p Holm = 5.72e-05
- **H1 regime** rapporto SD esterna/interna > 1: stima 2.846 [0.69, 11.78]; p = 0.0552; p Holm = 0.0731
- **H2 regime** ASL: ECE interna TS - Platt > 0: stima 0.08135 [0.0686, 0.0941]; p = 3.02e-05; p Holm = 0.000151
- **H3 swin** effetto ASL su energy-in-box: Grad-CAM - Grad-CAM++ != 0 (Grad-CAM +0.0208, Grad-CAM++ -0.1134): stima 0.1342 [0.0136, 0.2547]; p = 0.0366; p Holm = 0.0731
- **H4** cascata completa: rho(IG nero) - rho(Grad-CAM) > 0 (tutti i run): stima 0.5809 [0.554, 0.607]; p = 2.05e-41; p Holm = 1.43e-40
- **H5 regime** epoca selezionata: regime alternativo > ricetta del paper (Mann-Whitney): stima 2.5 ; p = 5.09e-05; p Holm = 0.000204
