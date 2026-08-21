\documentclass[preprint,12pt]{elsarticle}
%% ============================================================
%%  NOTE FOR THE AUTHOR / ADVISOR
%%  This file uses the standard Elsevier `elsarticle` class,
%%  which is preinstalled on Overleaf and in most TeX Live
%%  distributions. If compiling locally and the class is
%%  missing, download it from the CMIG "Guide for Authors"
%%  page on ScienceDirect (Elsevier author package).
%% ============================================================

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\biboptions{numbers,sort&compress}

\journal{Computerized Medical Imaging and Graphics}

\begin{document}

\begin{frontmatter}

\title{Beyond a Single Comparison: A Multi-Condition Audit of Attribution Localization, Calibration, and Robustness in Multi-Label Chest X-Ray Classification}

\author[vanvitelli]{Luca Migliaccio}
\author[vanvitelli]{Antonio Esposito\corref{cor1}}
\cortext[cor1]{Corresponding author.}
\ead{name.surname@unicampania.it}

\affiliation[vanvitelli]{organization={Department of Engineering, Universit\`a degli Studi della Campania ``Luigi Vanvitelli''},
            city={Aversa},
            country={Italy}}

\begin{abstract}
Deep learning models for automated chest radiograph interpretation are typically validated on discriminative accuracy alone, leaving open whether their predictions are well calibrated, robust to acquisition artifacts, and anatomically grounded. We present a multi-dimensional audit of multi-label thoracic disease classification on NIH ChestX-ray14, comparing eight backbone/pretraining configurations and two ensembles, then examining the best single backbone (ConvNeXt) along three further axes: an explainability-motivated preprocessing and training-schedule intervention targeting extra-thoracic shortcut artifacts, probability calibration, and robustness to synthetic corruption.

A single, uncontrolled before/after comparison of the intervention initially suggested a strong improvement in Grad-CAM localization against a body-silhouette proxy ($p=1.4\times10^{-28}$). A controlled seven-condition ablation with three seeds, a corrected mask measurement artifact, and a hierarchical bootstrap (partially accounting for having only three trained instances per condition) eliminated this effect entirely. Validating localization against NIH's official pathology bounding boxes, restricted to the 131 test-split images with annotations after correcting an initial train/test contamination bug, likewise found no robust improvement in pointing-game accuracy, though with limited statistical precision; a small decrease in activation mass inside the true lesion box was the only ground-truth interval excluding zero, an exploratory, uncorrected-for-multiplicity signal running counter to any improvement narrative.

Separately, reanalyzing the ablation's saved predictions found Asymmetric Loss associated with substantially worse native calibration than binary cross-entropy (roughly 29-fold higher Expected Calibration Error, concordant with Brier score and log-likelihood). This initially conflicted with a separately-computed single-instance estimate elsewhere in the paper; a diagnostic comparing checkpoints and re-evaluating under a unified pipeline reconciled the discrepancy and internally replicated the pattern, though the exact source of the earlier, superseded estimate was not isolated.

Discrimination is essentially unchanged by any factor (macro ROC-AUC 0.852--0.855) and degrades under synthetic noise and blur. We report this trajectory, an apparently strong effect dissolving under statistical and measurement scrutiny alongside a large calibration pattern that survived it, as the primary contribution: single, uncontrolled comparisons are an unreliable basis for debiasing claims in this setting, and the audit protocol itself, including the specific errors it caught, is offered as a transferable methodological case study.
\end{abstract}

\begin{keyword}
chest X-ray classification \sep deep learning \sep explainable AI \sep model calibration \sep robustness \sep multi-label learning \sep shortcut learning
\end{keyword}

\end{frontmatter}

%% ============================================================
\section{Introduction}
\label{sec:intro}
%% Adapted/translated from thesis Ch.1-2 (motivation) and Ch.7 intro.

Artificial intelligence is increasingly being integrated into radiological practice, with applications spanning triage, quantitative measurement, and diagnostic decision support \citep{dorsi2025future,almaitah2024role,velamala2025role}. Convolutional neural networks (CNNs) and, more recently, vision transformers have demonstrated strong discriminative performance on large-scale chest radiograph datasets \citep{litjens2017survey,shamshad2023transformers,takahashi2024comparison}. However, a growing body of evidence shows that high discrimination metrics alone are not sufficient evidence of clinical trustworthiness: models can achieve high area-under-the-curve scores while relying on spurious correlations unrelated to the underlying pathology, a failure mode known as \emph{shortcut learning} \citep{geirhos2020shortcut} and documented in real deployed chest-radiograph classifiers by \citet{zech2018variable}, and while producing probability estimates that are poorly calibrated to the true frequency of disease \citep{ovadia2019trust,rajaraman2022calibration}.

Despite over 80\% mean ROC-AUC being routinely reported on the NIH ChestX-ray14 benchmark by recent architectures \citep{li2025chexds,yanar2025comparative,fisher2025pretraining}, comparatively few studies report calibration, robustness to acquisition artifacts, and a quantitative (rather than purely visual) check for shortcut learning alongside discrimination metrics for the same model (Section~\ref{sec:relatedwork}). This gap motivates the evaluation protocol presented here.

In this paper we present an evaluation pipeline for multi-label thoracic disease classification that goes beyond discrimination metrics to jointly assess: (i) predictive performance across convolutional and transformer backbones under different pretraining regimes; (ii) the effect of an explainability-motivated preprocessing and training-schedule intervention on shortcut-learning indicators; (iii) probability calibration; and (iv) robustness under synthetic image degradation. Our contribution is not a novel architecture but a multi-dimensional exploratory audit protocol for a chest X-ray classifier along these dimensions, applied to a representative modern backbone (ConvNeXt); several of its components still require fixed seeds, matched controls, and released code before the protocol itself can be called fully reproducible (Section~\ref{sec:limitations}).

Concretely, this paper contributes:
\begin{itemize}
\item a systematic comparison of eight convolutional and transformer backbone/pretraining configurations, plus two ensemble configurations, for multi-label thoracic disease classification on NIH ChestX-ray14;
\item an explainability-motivated preprocessing and training-schedule intervention aimed at shortcut learning, assessed both qualitatively (1{,}000 Grad-CAM visual reports) and quantitatively (an aggregate body-silhouette localization metric, Section~\ref{sec:debiasing}), with confounds between preprocessing, loss, and schedule made explicit;
\item a calibration analysis via Expected Calibration Error and reliability diagrams at the per-class level; and
\item a robustness evaluation under synthetic noise and blur perturbations, jointly monitoring discrimination and predictive entropy.
\end{itemize}

%% ============================================================
\section{Related Work}
\label{sec:relatedwork}

\subsection{Benchmark performance on NIH ChestX-ray14}
The dataset used in this study was introduced by \citet{wang2017chestxray8} together with a ResNet-based baseline reaching a mean ROC-AUC of 0.745 across the 14 pathology classes. Shortly after, \citet{rajpurkar2017chexnet} introduced CheXNet, a fine-tuned DenseNet-121, which substantially improved per-class AUC and, on the pneumonia subtask, reportedly matched or exceeded practicing radiologists on the F1 metric. Incorporating explicit spatial priors, \citet{guendel2018location} proposed a location-aware DenseNet (DNetLoc) and reported a mean AUC of 0.81 on an enlarged, patient-wise-split cohort, while also showing that switching from a naive random split to a strict patient-level split lowers reported AUC by several points, a methodological caution directly relevant to the patient-level split protocol adopted in this study (Section~\ref{sec:dataset}). More recent work continues to push discrimination upward: an ensemble of DenseNet-121 and Swin Transformer (CheX-DS) reached a mean AUC of 0.838 \citep{li2025chexds}; a systematic comparison of CNN, transformer, and Mamba-based backbones under identical training conditions found a ConvFormer/hybrid configuration reaching 0.841 \citep{yanar2025comparative}; and a large-scale ensembling study combining multiple ConvNeXt pretraining variants recently reported a mean ROC-AUC of 0.940, the highest published to date on this dataset, while also documenting substantial patient overlap (67.4\% of the official validation split) in the dataset's originally released file-based split \citep{fisher2025pretraining}, again reinforcing the importance of the independently verified, zero-overlap patient-level split used in the present study (Section~\ref{sec:dataset}).

Table~\ref{tab:sota} positions our reference model against these benchmarks.

\begin{table}[htbp]
\centering
\caption{Macro-averaged ROC-AUC on NIH ChestX-ray14 (official 14-class task) reported in this work versus selected published results. Figures are as reported in the cited papers; splits, preprocessing, and exact class sets vary across studies and figures are not adjusted for this, so comparisons should be read as indicative rather than strictly controlled.}
\label{tab:sota}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcc}
\toprule
Method & Backbone & Macro ROC-AUC \\
\midrule
\citet{wang2017chestxray8} (2017) & ResNet & 0.745 \\
\citet{guendel2018location} (2018/19) & DenseNet (location-aware) & 0.81 \\
\citet{yanar2025comparative} (2025) & ConvFormer (hybrid) & 0.841 \\
\citet{li2025chexds} (2025) & DenseNet+SwinV2 ensemble & 0.838 \\
\textbf{This work, ConvNeXt (pre-intervention)} & \textbf{ConvNeXt} & \textbf{0.853} \\
\textbf{This work, ConvNeXt (post-intervention)} & \textbf{ConvNeXt} & \textbf{0.851} \\
\citet{fisher2025pretraining} (2025) & 3-model ConvNeXt ensemble & 0.940 \\
\bottomrule
\end{tabular}%
}
\end{table}

Our single-model ConvNeXt result is competitive with, and in several cases above, comparably-sized single-model or small-ensemble published results, while remaining below the largest recent multi-variant ensembles \citep{fisher2025pretraining}, which trade off simplicity and interpretability for an aggressive 8-model, 255-combination search that falls outside the scope of this work. The latter figure is drawn from a non-peer-reviewed preprint at the time of writing and should be read with that caveat.

\subsection{Shortcut learning and mitigation in chest radiography}
Beyond raw discrimination, a growing literature documents that chest radiograph classifiers frequently exploit spurious, non-pathological cues. In an early and influential case study, \citet{zech2018variable} showed that pneumonia classifiers trained across multiple hospital systems partly learned to detect hospital-specific imaging artifacts (e.g., portable-scanner markers) rather than pathology itself, causing large performance drops under external validation, a direct real-world instance of the shortcut-learning failure mode formalized by \citet{geirhos2020shortcut}. Formal mitigation strategies for this failure mode include re-weighting or auxiliary-classifier approaches such as \citet{nam2020learning}; the intervention evaluated in Section~\ref{sec:debiasing} of this work is a simpler, preprocessing- and training-schedule-level change rather than an implementation of any such published algorithm, a distinction we make explicit there. More recently, \citet{brown2023detecting} proposed a multitask-learning-based shortcut-testing protocol to directly probe whether a clinical model's unfairness across subgroups is attributable to shortcut learning, applying it to both radiology and dermatology tasks. Our aggregate localization metric (fraction of Grad-CAM activation mass falling inside an automatically estimated body silhouette, before vs.\ after the intervention) is a complementary, coarser-grained quantitative check in the same spirit, but does not test for shortcut learning tied to a specific sensitive attribute (e.g., hospital source, patient demographics) as in \citet{brown2023detecting} and \citet{zech2018variable}; this remains a natural extension (Section~\ref{sec:limitations}).

\subsection{Calibration in medical image classification}
Reliable predictive uncertainty is increasingly recognized as a prerequisite for safe deployment of black-box classifiers \citep{ovadia2019trust}. Expected Calibration Error, computed by binning predicted confidences and comparing them to empirical accuracy, is a standard tool for quantifying miscalibration in deep networks \citep[cf.][for the closely related Brier score]{brier1950verification}. Closest to our setting, \citet{rajaraman2022calibration} systematically studied the interaction between class imbalance, calibration method, and decision threshold on chest X-ray and fundus image classifiers, finding that calibration significantly improves performance at a fixed default threshold (0.5) but offers no significant benefit once thresholds are already tuned via a precision-recall curve, a finding that qualifies how the ECE results in Section~\ref{sec:calibration} of this work should be interpreted, since our reported metrics already use per-class Youden-optimal thresholds rather than a fixed 0.5 cutoff (see also Section~\ref{sec:discussion}).

\subsection{Deep architectures for medical image classification}
Convolutional architectures such as ResNet \citep{he2016deep}, DenseNet \citep{huang2017densely}, and EfficientNet \citep{tan2019efficientnet} have been widely adopted as backbones for chest radiograph classification, owing to strong transfer-learning performance from ImageNet pretraining \citep{russakovsky2015imagenet}. More recently, ConvNeXt \citep{liu2022convnet} and transformer-based architectures such as the Vision Transformer \citep{dosovitskiy2021image} and Swin Transformer V2 \citep{liu2022swinv2} have narrowed or surpassed the performance gap with CNNs on large-scale vision benchmarks, motivating their evaluation in the radiological domain \citep{takahashi2024comparison,yanar2025comparative}.

\subsection{Explainability techniques}
Post-hoc explainability techniques such as Grad-CAM \citep{selvaraju2017gradcam} and Eigen-CAM \citep{muhammad2020eigencam} are commonly used to visually audit the spatial basis of a model's prediction, and underpin both the qualitative panels (Section~\ref{sec:xai}) and the quantitative localization analysis (Section~\ref{sec:debiasing}) presented in this work.

%% ============================================================
\section{Materials and Methods}
\label{sec:methods}

\subsection{Dataset}
\label{sec:dataset}
%% Translated/adapted from thesis Ch.5.

We use the NIH ChestX-ray14 dataset \citep{wang2017chestxray8}, comprising frontal-view chest radiographs labeled for 14 thoracic pathologies via automated extraction from radiology reports. Figure~\ref{fig:examples} shows representative examples for each of the 14 pathology classes.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig1_dataset_examples.png}
\caption{Representative examples of the 14 thoracic pathology classes in the NIH ChestX-ray14 dataset. Red circles mark the approximate finding location and are derived directly from the dataset's official \texttt{BBox\_List\_2017.csv} annotation file \citep{wang2017chestxray8}, not manually added.}
\label{fig:examples}
\end{figure}

The release used comprises 112{,}120 frontal-view radiographs at native $1024\times1024$ resolution, associated with 30{,}805 unique patients. Splitting was performed at the patient level (not image level) with a fixed random seed (42) in a 70/15/15 proportion, yielding 21{,}563 patients in the training set and 4{,}621 patients in each of the validation and test sets; a formal intersection check confirmed zero patient overlap across the three partitions.

The class distribution exhibits substantial imbalance (Figure~\ref{fig:distribution}) and a long-tailed co-occurrence structure, since labels are not mutually exclusive (Figure~\ref{fig:cooccurrence}). The majority class, ``No Finding'', accounts for over 53\% of images; among the remaining multi-label cases, 20{,}796 images (18.55\% of the corpus) carry more than one simultaneous pathology label.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/fig2_class_distribution.png}
\caption{Frequency distribution of the 14 pathology classes in the dataset. Counts are not mutually exclusive due to the multi-label nature of the annotations.}
\label{fig:distribution}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.75\linewidth]{figures/fig3_cooccurrence.png}
\caption{Chord diagram of pairwise pathology co-occurrence. Band thickness is proportional to the frequency with which two pathologies co-occur in the same study.}
\label{fig:cooccurrence}
\end{figure}

Patient-level exclusivity was enforced across training, validation, and test splits (counts above) to prevent leakage from repeated studies of the same patient.

\subsection{Model architectures and training protocol}
\label{sec:architectures}
%% Adapted from thesis Ch.6.

We evaluated eight backbone/pretraining configurations: ResNet-50 \citep{he2016deep} and DenseNet-121 \citep{huang2017densely}, each under both ImageNet \citep{russakovsky2015imagenet} and radiology-specific (TorchXRayVision, \citealp{cohen2020torchxrayvision}) initialization (four configurations); EfficientNet-B7 \citep{tan2019efficientnet} under ImageNet-22k initialization (one configuration); Swin Transformer V2 \citep{liu2022swinv2} under both ImageNet-22k and radiology-pretrained initialization (two configurations); and ConvNeXt \citep{liu2022convnet} under ImageNet-22k initialization (one configuration). Two further, separate configurations (ensembles combining (a) DenseNet-121 and ResNet-50, both radiology-pretrained, and (b) ConvNeXt and radiology-pretrained Swin Transformer V2) were evaluated alongside these eight single-backbone configurations but are not counted among them. All models were adapted for multi-label classification via a sigmoid output layer, trained with an asymmetric loss formulation to address class imbalance \citep{ridnik2021asymmetric,tsoumakas2007multilabel}.

Figure~\ref{fig:comparison} summarizes macro-averaged ROC-AUC and macro-averaged Recall, computed from continuous model scores prior to any operating-threshold selection, for the four best-discriminating configurations; Table~\ref{tab:allmodels} reports the same metrics for all eight single-backbone configurations plus the two ensemble configurations evaluated in this study, to avoid the selection bias of showing only a subset.

\begin{table}[htbp]
\centering
\caption{Global performance of all evaluated configurations. Macro/Micro ROC-AUC and PR-AUC are computed from continuous model scores and do not depend on a decision threshold; Precision, Recall, and F1 use the default 0.5 sigmoid threshold (per-class Youden-optimal thresholds are used only for the reference model in Section~\ref{sec:threshold} onward). The two ensemble rows are separate configurations, evaluated in addition to (not among) the eight single-backbone configurations.}
\label{tab:allmodels}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Configuration & Macro AUC & Micro AUC & PR-AUC & Precision & Recall & F1 \\
\midrule
ResNet-50, ImageNet            & 0.8116 & 0.8308 & 0.2049 & 0.2248 & 0.3564 & 0.2689 \\
ResNet-50, TorchXRayVision     & 0.8444 & 0.8814 & 0.2643 & 0.2789 & 0.4085 & 0.3153 \\
DenseNet-121, ImageNet         & 0.8424 & 0.8820 & 0.2760 & 0.2755 & 0.4381 & 0.3292 \\
DenseNet-121, TorchXRayVision  & 0.8366 & 0.8790 & 0.2563 & 0.2762 & 0.3992 & 0.3197 \\
EfficientNet-B7, ImageNet-22k  & 0.8427 & 0.8828 & 0.2572 & 0.1217 & 0.8163 & 0.1777 \\
SwinV2, ImageNet-22k           & 0.8490 & 0.8800 & 0.2811 & 0.1256 & 0.7834 & 0.2117 \\
SwinV2, radiology-pretrained   & 0.8533 & 0.8628 & 0.2764 & 0.1500 & 0.7298 & 0.2343 \\
\textbf{ConvNeXt, ImageNet-22k (reference)} & \textbf{0.8533} & \textbf{0.8891} & \textbf{0.2889} & \textbf{0.1468} & \textbf{0.7672} & \textbf{0.2322} \\
\midrule
Ensemble: DenseNet-121+ResNet-50 (both TorchXRayVision) & 0.8390 & 0.8509 & 0.2478 & 0.1342 & 0.7380 & 0.2150 \\
Ensemble: ConvNeXt+SwinV2 (radiology-pretrained)         & 0.8563 & 0.8703 & 0.2864 & 0.1474 & 0.7695 & 0.2330 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/fig4_model_comparison.png}
\caption{Comparison of macro ROC-AUC and macro Recall across the top four model configurations (default 0.5 threshold, pre-threshold-optimization). Axis and legend labels in the source figure are in Italian; values correspond to the macro ROC-AUC and macro Recall metrics defined in the text.}
\label{fig:comparison}
\end{figure}

Based on this comparison and on the qualitative interpretability analysis in Section~\ref{sec:xai}, ConvNeXt (ImageNet-22k pretraining) was selected as the reference model for the remainder of the study. This choice is not the top-AUC configuration in absolute terms: the ConvNeXt--SwinV2 ensemble reaches a marginally higher macro ROC-AUC (0.8563 vs.\ 0.8533), but ConvNeXt was preferred on Grad-CAM interpretability grounds: attention-based backbones produced activation maps with higher-frequency artifacts, spatial fragmentation, and color-shift discontinuities relative to the anatomically continuous maps produced by the purely convolutional ConvNeXt. We flag that this selection criterion was a visual judgment made by the authors during model development, without a predefined scoring rubric, blinded assessment, or inter-rater agreement check, and that a smoother-looking activation map is not itself evidence of a more faithful explanation, since smoothness can also result from spatial resolution or the choice of target layer rather than from more anatomically grounded reasoning. This is a methodological limitation of the model-selection step (Section~\ref{sec:limitations}), and because ConvNeXt was then carried forward into the explainability and debiasing analysis that motivated its own selection, the reasoning has an element of circularity that a predefined, blinded selection protocol would avoid in future work.

\subsection{Decision threshold optimization}
\label{sec:threshold}
Per-class decision thresholds were optimized using the Youden index \citep{youden1950index}, defined as $J = \text{sensitivity} + \text{specificity} - 1$ and maximized over the ROC curve; this balances sensitivity and specificity but does not explicitly maximize recall, incorporate disease prevalence, or encode the relative clinical costs of false negatives versus false positives. For both the pre- and post-intervention ConvNeXt models, the Youden-optimal threshold for each class was computed exclusively from validation-split predictions, and the resulting fixed thresholds were then applied, without refitting, to model predictions on the held-out test split for all metrics reported in Table~\ref{tab:debiasing} and Table~\ref{tab:perclass}; the same validation-fit, test-evaluated protocol was used for the calibration and robustness analyses. There is therefore no data leakage between threshold selection and performance reporting.

\subsection{Explainability-motivated mitigation intervention: a controlled multi-condition ablation}
\label{sec:debiasing}

Preliminary Grad-CAM \citep{selvaraju2017gradcam} inspection of the reference model revealed a shortcut-learning pattern \citep{geirhos2020shortcut}: activation was frequently concentrated outside the thoracic cavity, on radiographic markers, positioning artifacts, or image borders, rather than on lung parenchyma (Figure~\ref{fig:shortcut}).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\linewidth]{figures/fig5_shortcut_learning.png}
\caption{Example of shortcut learning: an activation map showing spurious activation on peripheral markers and image borders rather than on the lung parenchyma. This pattern motivated the intervention studied below.}
\label{fig:shortcut}
\end{figure}

An earlier version of this analysis retrained ConvNeXt with three simultaneous changes (input cropping, loss function, and training schedule) and reported a single before/after comparison. Because these three factors were confounded, and because a single trained instance cannot separate a genuine effect from ordinary training variation, we replaced that comparison with a controlled multi-condition ablation with three independent random seeds (42, 123, 2026) per condition. This is not a full factorial design: with four candidate factors a complete factorial would require $2^4=16$ conditions, and we ran seven (a baseline, one arm changing each factor individually, one two-factor combination, and the full four-factor combination), so main effects and interactions beyond the specific combinations tested are not systematically estimable. The seven conditions are:

\begin{itemize}
\item \textbf{C0 (baseline)}: direct resize to $384\times384$, Asymmetric Loss \citep{ridnik2021asymmetric}, random horizontal flip augmentation, fixed 10 epochs with cosine-annealing learning rate.
\item \textbf{C1 (crop only)}: as C0, but resize to $420\times420$ followed by a center crop to $384\times384$.
\item \textbf{C2 (loss only)}: as C0, but \texttt{BCEWithLogitsLoss} instead of Asymmetric Loss.
\item \textbf{C3 (schedule only)}: as C0, but up to 20 epochs with early stopping (patience 8) and no learning-rate scheduler.
\item \textbf{C4 (augmentation removed)}: as C0, but without random horizontal flip.
\item \textbf{C5 (crop + loss)}: C1 and C2 combined.
\item \textbf{C6 (full combination)}: all four changes combined (equivalent to the originally described ``intervention'').
\end{itemize}

All other hyperparameters (Table~\ref{tab:hyperparams}) were held fixed across conditions, including weight decay ($10^{-2}$, the explicit pre-intervention value and also PyTorch's AdamW default, so not a genuine third value as an earlier draft of this analysis suggested).

\begin{table}[htbp]
\centering
\caption{Hyperparameters held fixed across all seven ablation conditions, and the four factors that vary (see level definitions in the text above).}
\label{tab:hyperparams}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ll}
\toprule
Setting & Value (fixed across all conditions) \\
\midrule
Backbone \& checkpoint & \texttt{convnext\_base.fb\_in22k} (timm), ImageNet-22k pretrained \\
Optimizer & AdamW, weight decay $10^{-2}$ \\
Learning rate (initial) & $3\times10^{-5}$ \\
Batch size & 16 \\
Checkpoint selection & Best macro ROC-AUC on the validation split \\
Random seeds (independent runs per condition) & 42, 123, 2026 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsubsection{Global performance across conditions}

Table~\ref{tab:debiasing} reports macro ROC-AUC and macro PR-AUC, mean $\pm$ standard deviation across the three seeds, for all seven conditions. No condition differs from the baseline by more than the seed-to-seed noise: macro ROC-AUC ranges from 0.8520 to 0.8548 across all seven conditions and 21 runs, with per-condition standard deviations between 0.0004 and 0.0023, an order of magnitude smaller than any of the pairwise differences. \textbf{None of the four factors, alone or in combination, produces a discrimination change distinguishable from training noise.}

\begin{table}[htbp]
\centering
\caption{Global discrimination performance across the seven ablation conditions (this table does not include Precision, Recall, or F1), mean $\pm$ standard deviation over 3 independent random seeds each. Full precision/recall/F1 breakdowns per condition and per seed are released as CSV files alongside this work (see the Data Availability statement at the end of this article) rather than reproduced here for space.}
\label{tab:debiasing}
\begin{tabular}{lcc}
\toprule
Condition & Macro ROC-AUC & Macro PR-AUC \\
\midrule
C0 baseline            & $0.8537 \pm 0.0007$ & $0.2777 \pm 0.0054$ \\
C1 crop only            & $0.8545 \pm 0.0023$ & $0.2737 \pm 0.0048$ \\
C2 loss only             & $0.8520 \pm 0.0020$ & $0.2787 \pm 0.0057$ \\
C3 schedule only         & $0.8535 \pm 0.0007$ & $0.2790 \pm 0.0048$ \\
C4 augmentation removed  & $0.8526 \pm 0.0014$ & $0.2753 \pm 0.0040$ \\
C5 crop + loss           & $0.8539 \pm 0.0004$ & $0.2783 \pm 0.0066$ \\
C6 full combination      & $0.8538 \pm 0.0010$ & $0.2714 \pm 0.0046$ \\
\bottomrule
\end{tabular}
\end{table}

The very low absolute Macro Precision reported for the reference single-instance model in Section~\ref{sec:results} (0.1468 pre-intervention, 0.1502 post-intervention) is not an effect of the intervention: precision was already low under the pre-intervention Youden-calibrated thresholds and remained essentially unchanged post-intervention. At the Youden-selected thresholds, sensitivity was relatively high but positive predictive value was low; this is a known consequence of thresholding for sensitivity/specificity balance under low class prevalence, where even a well-discriminating model (high ROC-AUC) yields many false positives per true positive once converted to a binary decision at low prevalence.

\subsubsection{A mask-measurement artifact discovered during visual quality control}

Before reporting the localization comparison, we describe a methodological correction that materially changed our conclusions and that we report for transparency. The localization metric quantifies the fraction of Grad-CAM activation mass falling inside a body-silhouette mask (Otsu thresholding on the radiograph, largest connected component, then erosion to exclude boundary pixels). A visual audit of the images with the largest apparent condition-vs-baseline differences revealed that this mask frequently \emph{excluded the patient's shoulder}: because X-ray intensity often dips at the shoulder joint, the shoulder is occasionally disconnected from the main torso blob under simple thresholding, and taking only the largest connected component then discards it as ``not body,'' even though it is genuine patient anatomy rather than background or an artifact. Any condition whose Grad-CAM attended more to the shoulder/clavicle region than another would be mechanically penalized by this mask defect, independent of whether either behavior is more or less appropriate. We corrected this by adding a morphological closing operation (disk radius 20\,px) before extracting the largest connected component, which bridges small intensity gaps at joints while leaving genuine background separated; a direct visual comparison confirmed the corrected mask includes the shoulder where the original mask excluded it. All results reported below use this corrected mask; applying it uniformly raised the mean in-body fraction from approximately 0.59--0.62 to approximately 0.74--0.76 across every condition (a comparable shift for all conditions, consistent with a shared measurement artifact rather than a condition-specific effect), and, as detailed next, substantially narrowed the apparent gap between conditions that we had observed with the uncorrected mask.

\subsubsection{Localization results}

Table~\ref{tab:localization} reports, for each condition against the C0 baseline over the same $N=1{,}000$ test-set sample (768 unique patients) used throughout this study, the median paired difference, interquartile range (IQR) of paired differences, percentage of images that improved, a patient-only bootstrap 95\% confidence interval (resampling by patient rather than by image, since a patient may contribute more than one study), a hierarchical bootstrap 95\% confidence interval (resampling by seed, then by patient within each resampled seed), and a paired Wilcoxon signed-rank $p$-value.

\begin{table}[htbp]
\centering
\caption{Localization metric (fraction of Grad-CAM activation inside the corrected body-silhouette mask), each condition vs.\ the C0 baseline, over $N=3{,}000$ paired observations (1{,}000 images $\times$ 3 seeds, 768 unique patients). PATIENT CI: bootstrap resampling patients only (treats each seed's images as additional independent observations per patient). HIER.\ CI: hierarchical bootstrap resampling seeds first, then patients within each resampled seed; this is the statistically appropriate interval given that only 3 independent model instances were trained per condition. 10{,}000 resamples each.}
\label{tab:localization}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Condition & Median $\Delta$ & IQR & \% improved & Patient-only 95\% CI & Hierarchical 95\% CI & Wilcoxon $p$ \\
\midrule
C1 crop only            & $\approx 0$     & 0.059 & 46.7\% & $[-0.0063, +0.0008]$ & $[-0.0064, +0.0009]$ & 0.21 \\
C2 loss only             & $+0.0032$        & 0.090 & 56.4\% & $[+0.0083, +0.0183]$ & $[-0.0008, +0.0299]$ & $1.8\times10^{-22}$ \\
C3 schedule only         & $\approx 0$     & 0.010 & 42.1\% & $[-0.0045, -0.0013]$ & $[-0.0062, +0.0001]$ & $2.1\times10^{-15}$ \\
C4 augmentation removed  & $\approx 0$     & 0.052 & 50.3\% & $[-0.0046, +0.0031]$ & $[-0.0068, +0.0045]$ & 0.054 \\
C5 crop + loss            & $-0.0016$        & 0.111 & 44.4\% & $[-0.0206, -0.0090]$ & $[-0.0361, +0.0011]$ & $4.2\times10^{-7}$ \\
C6 full combination      & $0.0000$        & 0.102 & 47.4\% & $[-0.0171, -0.0057]$ & $[-0.0337, +0.0067]$ & 0.0063 \\
\bottomrule
\end{tabular}%
}
\end{table}

Three findings stand out, and the hierarchical CI changes the headline conclusion relative to the patient-only CI reported in an earlier version of this analysis. First, \textbf{under the patient-only bootstrap, four of six conditions (C2, C3, C5, C6) appeared to have confidence intervals excluding zero; under the hierarchical bootstrap, which additionally accounts for the fact that only three independent model instances were trained per condition, all six conditions' intervals include zero}, C3 only barely (upper bound $+0.0001$). This is the more statistically appropriate result: with three seeds, an effect must be large and consistent across all three trained instances to be distinguishable from ordinary training variability, and none of the six conditions meets that bar. Second, \textbf{no individual factor produces a change that is statistically distinguishable from zero under the hierarchical treatment}; C2 (loss) retains the largest point estimate and the most extreme $p$-value, but even its hierarchical interval spans from essentially zero to a modest positive value. Third, \textbf{the practical size of every effect is small regardless of statistical treatment}: median paired differences are at or near zero for five of the six conditions, interquartile ranges are wide (0.05--0.11) relative to the medians, and the percentage of images that improve is close to 50\% in every condition (42--56\%), indicating no systematic, model-wide relocation of attention in either direction.

A statistical caveat applied to an earlier version of this analysis, which reported only a patient-only bootstrap: the Wilcoxon $p$-values and the patient-only CI both treat all 3{,}000 paired observations as independent, but they derive from only three independent trained instances (seeds) per condition, evaluated on the same 1{,}000 images each. This is a source of pseudoreplication that neither the Wilcoxon test nor the patient-only bootstrap models, and it made the smallest $p$-values (e.g., C2's $1.8\times10^{-22}$) and the patient-only intervals considerably more confident than the true evidence against the null warrants. We addressed this directly with the hierarchical bootstrap reported in Table~\ref{tab:localization} (resampling seeds, then patients within each resampled seed), which is the statistically appropriate treatment and the one we rely on for the interpretation above; we retain the $p$-values and patient-only CI in the table for transparency and comparison, but they should not be read as the primary evidence. With only three seeds, the hierarchical bootstrap's seed-level resampling is itself statistically coarse (27 possible seed combinations); training two additional seeds (five total) would sharpen these intervals further, which we did not do here.

We conclude that this multi-condition ablation does not support a claim that the studied intervention, under any of its individual components or their combination, systematically improves Grad-CAM localization relative to the body-silhouette mask, once measurement artifacts and training-seed variance are properly accounted for. The single-run comparison in an earlier version of this analysis (mean $0.556\to0.601$, $p=1.4\times10^{-28}$) reflected a combination of the shoulder-exclusion mask artifact described above and ordinary seed-to-seed variation, not a robust effect of the intervention; we report this explicitly as a demonstration of why single, uncontrolled before/after comparisons are an unreliable basis for debiasing claims in this setting, which we regard as an independently useful finding of this work (Section~\ref{sec:discussion}).

\textbf{Grad-CAM implementation details.} For reproducibility: the target layer is the final stage of the ConvNeXt backbone (\texttt{model.stages[-1]}), identical across all seven conditions; the target class for each image is that image's own top-1 predicted class under the condition being evaluated (i.e., not a fixed class across conditions, and not evaluated separately per ground-truth label for multi-label images); activation maps are the standard \texttt{pytorch-grad-cam} \texttt{GradCAM} output, clipped to non-negative values before computing the in-mask fraction (negative activation is excluded from the numerator and denominator alike, rather than being treated as evidence against the class); maps are computed at, and the body mask is rasterized at, the model's native $384\times384$ input resolution (no additional upsampling to original image resolution); and the metric is computed identically regardless of whether the top-1 prediction corresponds to a true positive, false positive, or negative case, i.e., no stratification by prediction correctness is applied in the headline numbers above.

\subsubsection{Validation against ground-truth pathology localization}
\label{sec:bbox}

The body-silhouette metric above, even corrected, cannot distinguish activation on the lung parenchyma from activation on the heart, ribs, spine, or other non-pulmonary structures that also fall inside the body outline: it is a proxy for ``not an extra-thoracic artifact,'' not a measure of pathology localization. NIH ChestX-ray14 includes an official set of 984 manually annotated bounding boxes (\texttt{BBox\_List\_2017.csv}) covering 880 unique images and 8 of the 14 pathology classes, the same annotations underlying Figure~\ref{fig:examples}; \textbf{restricting this set to images in our held-out test split (essential to avoid evaluating localization on images the model may have seen during training, even though the boxes themselves were never used as training supervision) leaves 146 annotations on 131 unique test-set images}, a substantially smaller sample than the 1{,}000-image draw used for the body-silhouette metric. An earlier version of this analysis mistakenly used the full 984-annotation set without this filter; we correct that here and report only the properly test-restricted result. On this set, for C0 and C6 (the reference comparison), each over 3 seeds, we compute two standard weakly-supervised localization metrics against true pathology location: \emph{pointing-game accuracy} (does the pixel of maximum Grad-CAM activation fall inside the annotated box, for the image's ground-truth class rather than its top-1 prediction), and \emph{energy-in-box} (the fraction of Grad-CAM activation mass falling inside the box, directly analogous to the body-silhouette metric but against real pathology location).

\begin{table}[htbp]
\centering
\caption{Ground-truth pathology localization, C6 vs.\ C0, restricted to the 131 unique test-split images with official bounding-box annotations (146 annotations $\times$ 3 seeds, $N=438$ paired observations, 113 unique patients), correctly paired by image and annotated class. The hierarchical bootstrap CI (seed then patient, 10{,}000 resamples) is the primary evidence in this table; the Wilcoxon $p$-value is reported as a secondary, descriptive statistic and, like the CI, treats the underlying observations as more independent than the seed-clustered design strictly allows (Section~\ref{sec:limitations}).}
\label{tab:bbox}
\begin{tabular}{lccccc}
\toprule
Metric & Mean $\Delta$ & Median $\Delta$ & \% improved & Hierarchical 95\% CI & Wilcoxon $p$ \\
\midrule
Pointing-game accuracy & $+0.030$ & $0.000$ & 8.9\% & $[-0.011, +0.071]$ & 0.107 \\
Energy-in-box           & $-0.008$ & $-0.010$ & 42.2\% & $[-0.0165, -0.0004]$ & $9.2\times10^{-4}$ \\
\bottomrule
\end{tabular}
\end{table}

Raw pointing-game accuracy is 0.527--0.555 for C0 and 0.541--0.582 for C6 across the three seeds; the point estimate still favors C6 in every seed pairing, but \textbf{once restricted to genuinely held-out test images, this effect no longer survives the hierarchical bootstrap}: the 95\% CI spans from a small negative to a moderate positive value and includes zero. We quantify the precision limitation explicitly rather than only asserting it, via a sensitivity calculation rather than an exact power analysis (the standard formula for the minimum detectable difference between two proportions at 80\% power and $\alpha=0.05$ is an unpaired approximation; our design is paired and pools three seeds per image, and we do not have the discordant-pair correlation structure needed for an exact calculation under that design, so this should be read as an approximate, conservative bound rather than a precise power calculation): under this approximation, $n=131$ images provides adequate sensitivity to detect a difference of roughly 17 percentage points or more, nearly six times the $+3$-percentage-point effect point-estimated here. We therefore cannot distinguish between ``the effect is real but this sample is too small to confirm it'' and ``the effect is not real and the earlier, larger-sample estimate (using data that included train/validation images) was itself inflated by the same kind of uncontrolled comparison this paper otherwise warns against.'' We do not adjudicate between these and report the null result as the honest outcome of the properly restricted analysis; a larger externally annotated dataset would be needed to resolve this with confidence.

The energy-in-box result is the only ground-truth localization interval in this study excluding zero after correcting for the train/test contamination and applying the hierarchical bootstrap, though the upper bound ($-0.0004$) is close enough to zero that we treat this as a fragile, exploratory signal rather than a confirmed effect: we tested two localization metrics (pointing-game and energy-in-box) without a pre-specified primary endpoint or a multiple-comparisons correction, so a single borderline interval among several tests should not be read as confirmatory on its own. Its direction (C6 places slightly \emph{less} total activation mass inside the true pathology box than C0, not more) runs counter to any narrative in which the intervention improves pathology localization; if anything, it is mildly discouraging for that narrative, though the effect is too small and too marginal statistically to support a strong claim in either direction.

\textbf{Handling of multiple annotations.} Some images carry more than one bounding-box annotation (either multiple instances of the same pathology or annotations for different pathologies); we did not merge or deduplicate these. Each (image, annotated class) pair is treated as a separate analysis row, later clustered by patient in the hierarchical bootstrap rather than treated as statistically independent, with its own Grad-CAM computation targeting that specific ground-truth class, consistent with the ``146 annotations on 131 images'' counts reported above; an image with two annotated classes therefore contributes two rows (and, after pairing with the corresponding C0/C6 predictions, two paired observations) rather than one. We did not encounter multiple annotations of the \emph{same} class on the same image in this test-restricted subset, so the question of merging overlapping same-class boxes did not arise here, but we note it as a rule future extensions of this analysis (e.g., to the intermediate ablation conditions) should specify explicitly. The hierarchical bootstrap resamples at the seed level, then the patient level, exactly as for the body-silhouette metric (Section~\ref{sec:debiasing}); a patient with multiple annotated images or classes contributes all of their associated observations together whenever that patient is selected in a bootstrap resample.

This comparison was run only for C0 and C6, not the intermediate ablation conditions, and only on a small, officially-annotated test subset; extending it to the other five conditions and, ideally, to a larger externally-annotated set would be needed before drawing firmer conclusions about any specific factor.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\linewidth]{figures/fig6_debiased_gradcam.png}
\caption{A single Effusion case where the C6 configuration's activation is more concentrated within the body silhouette than C0's, shown only to illustrate what the qualitative reports in Section~\ref{sec:xai} look like. Given the multi-condition results above, this example is not representative of a systematic effect and should not be read as evidence of one; roughly half of the test images show a shift in the opposite direction.}
\label{fig:debiased}
\end{figure}

\subsection{Qualitative interpretability analysis}
\label{sec:xai}
To complement the aggregate localization analysis, we generated multi-panel visual reports combining the original radiograph, its Grad-CAM activation map, and the full per-class probability spectrum, for a random sample of 1{,}000 test-set images (fixed seed 42), for representative True Positive, True Negative, False Positive, and False Negative cases.

\subsection{Calibration analysis}
\label{sec:calibration}
%% Adapted from thesis \S7.3.

Model calibration was quantified via the Expected Calibration Error \citep{guo2017calibration}, computed independently for each of the 14 classes using 10 uniform-width confidence bins (\texttt{sklearn.calibration.calibration\_curve}, \texttt{strategy='uniform'}) over the model's raw sigmoid outputs (interpreted as the predicted probability of the positive class). Following the standard definition, for $B$ bins each containing $n_b$ of the $N$ test samples, with $\text{acc}(b)$ and $\text{conf}(b)$ the empirical accuracy and mean predicted confidence within bin $b$:
\[
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|.
\]
Bins with zero samples are excluded from the sum (equivalently, treated as contributing zero weight). No post-hoc calibration method (e.g., temperature scaling, Platt scaling, or isotonic regression) was applied; the analysis characterizes the model's native, uncalibrated probability outputs, and we do not report bin-level sample counts, confidence intervals, or complementary metrics (Brier score, negative log-likelihood, calibration slope/intercept) for this specific single-instance breakdown (Section~\ref{sec:calibration-ablation} reports these for the ablation as a whole). \textbf{The per-class values below were recomputed after we identified and corrected a discrepancy between two evaluation pipelines (see Section~\ref{sec:calibration-ablation} and the note on Figure~\ref{fig:reliability})}; an earlier version of this analysis, using a since-superseded evaluation pipeline, had reported a mean ECE of 0.1015 for this same checkpoint, roughly 10-fold higher than the value below; a diagnostic comparison (released with this work, \texttt{diagnose\_ece\_discrepancy.py}) confirmed this was an artifact of the earlier pipeline, not of the model, by reproducing the higher figure's discrepancy and tracing it away once the checkpoint was re-evaluated under the same preprocessing and evaluation code used consistently elsewhere in this paper.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\linewidth]{figures/fig7_reliability_diagram.png}
\caption{Reliability diagram for the Effusion class, generated under the corrected evaluation pipeline (Section~\ref{sec:calibration-ablation}) using \texttt{regenerate\_reliability\_diagram.py}, released with this work. The dashed diagonal indicates perfect calibration; the solid curve shows the empirical relationship between predicted confidence and observed accuracy, with the number of test samples in each confidence bin ($n$) shown alongside each point. ECE for this class is 0.0158, matching Table~\ref{tab:ece}; this supersedes an earlier version of this figure (ECE 0.0530, Italian axis labels) generated under the since-superseded evaluation pipeline discussed in Section~\ref{sec:calibration-ablation}. The curve lies close to the diagonal across bins with substantial sample sizes (up to $n=12{,}356$ in the lowest-confidence bin), with the sparsest bins ($n \leq 86$) at the highest confidences contributing more to the residual ECE.}
\label{fig:reliability}
\end{figure}

The corrected mean ECE across all 14 classes is 0.0106, with substantial relative per-class variability (range: 0.0006 for Hernia to 0.0402 for Infiltration) even though all values are now an order of magnitude smaller than originally reported, as detailed in Table~\ref{tab:ece}.

\begin{table}[htbp]
\centering
\caption{Expected Calibration Error (ECE) per pathology class, sorted by increasing miscalibration. Corrected values, recomputed under the same evaluation pipeline used throughout this paper (see text); superseded the previously reported 0.1015 mean.}
\label{tab:ece}
\begin{tabular}{lc}
\toprule
Pathology & ECE \\
\midrule
Hernia              & 0.0006 \\
Cardiomegaly        & 0.0029 \\
Emphysema           & 0.0037 \\
Pneumonia           & 0.0041 \\
Fibrosis            & 0.0060 \\
Mass                & 0.0061 \\
Edema               & 0.0068 \\
Consolidation       & 0.0079 \\
Pleural Thickening  & 0.0089 \\
Atelectasis         & 0.0118 \\
Effusion            & 0.0158 \\
Pneumothorax        & 0.0163 \\
Nodule              & 0.0169 \\
Infiltration        & 0.0402 \\
\midrule
\textbf{Mean (macro)} & \textbf{0.0106} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Calibration across ablation conditions: a loss-function-driven effect, now reconciled}
\label{sec:calibration-ablation}

The per-class breakdown above is for a single trained instance of the reference (post-intervention) configuration, now consistent with the ablation's own conditions (below) to within seed-level noise. Reanalyzing the saved test-set predictions from all seven ablation conditions (Section~\ref{sec:debiasing}) across all three seeds each (computing macro ECE, Brier score, and negative log-likelihood (NLL) for every condition/seed, with no new inference required) revealed by far the largest and most consistent pattern in this entire study, and one we had not anticipated: \textbf{loss function is associated with calibration quality far more strongly than any localization-related factor}. We initially could not reconcile this internal ablation pattern with the single reference instance's separately-computed per-class breakdown above (which, before correction, reported a mean nearly 10-fold higher); a targeted diagnostic (comparing checkpoint weights directly, then re-evaluating the reference checkpoint under the ablation's own evaluation pipeline) reconciled the two and internally replicated the loss-associated pattern, though it did not isolate the exact implementation difference in the earlier pipeline responsible for its higher estimate (Section~\ref{sec:limitations}).

\begin{table}[htbp]
\centering
\caption{Calibration metrics across ablation conditions, mean $\pm$ standard deviation over 3 seeds. Conditions are grouped by loss function; the resulting pattern is now confirmed consistent with the single-instance reference model's corrected calibration (Table~\ref{tab:ece}, Section~\ref{sec:calibration-ablation}). Single-seed patient-clustered bootstrap CIs for macro ECE, and the full per-seed breakdown, are released with this work (Data Availability) rather than shown inline here, to avoid the ambiguity of juxtaposing a 3-seed mean with a single-seed interval.}
\label{tab:calib-ablation}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccc}
\toprule
Loss function & Condition & Macro Brier (3-seed) & Macro NLL (3-seed) & Macro ECE (3-seed) \\
\midrule
\multirow{4}{*}{Asymmetric Loss \citep{ridnik2021asymmetric}}
 & C0 baseline            & $0.1261 \pm 0.0040$ & $0.4268 \pm 0.0095$ & $0.2892 \pm 0.0068$ \\
 & C1 crop only           & $0.1264 \pm 0.0032$ & $0.4279 \pm 0.0076$ & $0.2901 \pm 0.0054$ \\
 & C3 schedule only       & $0.1259 \pm 0.0037$ & $0.4264 \pm 0.0087$ & $0.2888 \pm 0.0061$ \\
 & C4 augmentation removed & $0.1253 \pm 0.0070$ & $0.4251 \pm 0.0168$ & $0.2880 \pm 0.0121$ \\
\midrule
\multirow{3}{*}{\texttt{BCEWithLogitsLoss}}
 & C2 loss only           & $0.0383 \pm 0.0004$ & $0.1399 \pm 0.0012$ & $0.0099 \pm 0.0013$ \\
 & C5 crop + loss          & $0.0384 \pm 0.0001$ & $0.1400 \pm 0.0006$ & $0.0106 \pm 0.0016$ \\
 & C6 full combination    & $0.0383 \pm 0.0004$ & $0.1398 \pm 0.0011$ & $0.0101 \pm 0.0015$ \\
\bottomrule
\end{tabular}%
}
\end{table}

The separation is consistent across the evaluated runs: every condition using Asymmetric Loss has a macro ECE of approximately 0.29, and every condition using \texttt{BCEWithLogitsLoss} has a macro ECE of approximately 0.01, with seed-to-seed standard deviations (0.0004--0.012) two orders of magnitude smaller than the gap between the two groups. Brier score and NLL show the same pattern (Brier $\approx 0.126$ vs.\ $\approx 0.038$; NLL $\approx 0.43$ vs.\ $\approx 0.14$), and the concordance across three independently-computed metrics makes it unlikely that this pattern is solely an artifact of ECE's known sensitivity to bin choice. Crop, schedule, and augmentation have no visible effect on any calibration metric within either loss-function group. The direction is unsurprising given how the two losses are constructed: binary cross-entropy (log loss) is a strictly proper scoring rule \citep{gneiting2007strictly}, meaning it is minimized in expectation only when predicted probabilities match the true underlying probabilities, which is precisely the condition for good calibration; Asymmetric Loss \citep{ridnik2021asymmetric} deliberately departs from this by reweighting the contribution of easy negatives via its $\gamma_-=4$ focusing term to improve ranking-based discrimination under severe class imbalance, at the cost of the probabilistic interpretation a proper scoring rule guarantees; we report this pattern as an association consistent with that mechanism, not as an isolated causal effect independent of the other confounded changes in each condition (Section~\ref{sec:debiasing}). Re-evaluating the single reference instance under this ablation's unified evaluation pipeline reconciled its calibration estimate with the ablation's own BCE-based conditions (corrected mean ECE 0.0106, Table~\ref{tab:ece}, versus C2: 0.0099, C5: 0.0106, C6: 0.0101) and internally replicated the loss-associated calibration pattern across conditions and seeds; the exact implementation difference in the earlier, superseded pipeline responsible for its higher estimate (0.1015) remains unidentified (Section~\ref{sec:limitations}), so we describe this reconciliation as an internal replication rather than an independent validation.

This finding is largely orthogonal to the localization question that motivated the ablation, and we did not set out to test it; we report it because it emerged directly from the same experimental infrastructure and is, by a wide margin, the largest and most consistent pattern observed in this study.

\subsection{Robustness analysis}
\label{sec:robustness}
%% Adapted from thesis \S7.4.

To assess resilience to acquisition artifacts, we evaluated the reference model under two synthetic perturbations applied to the test set at a single severity level each: additive Gaussian noise with standard deviation 0.05, applied to the tensor-converted image prior to normalization; and Gaussian blur with kernel size 7 applied to the image prior to tensor conversion, using torchvision's default sigma sampling (uniform in $[0.1, 2.0]$ per image, i.e., blur severity varies across the test set rather than being fixed). Both the clean baseline and the corrupted scenarios in this experiment use direct resizing to $384\times384$ without center-cropping, which differs from the $420\to384$ resize-then-crop pipeline that the post-intervention model is otherwise evaluated with elsewhere in this study (Sections~\ref{sec:threshold}--\ref{sec:calibration}); because all three scenarios here (clean, noise, blur) use the same preprocessing as each other, the noise-versus-blur-versus-clean comparison within this table is internally consistent, but the clean-baseline row in Table~\ref{tab:robustness} (ROC-AUC 0.8512) is not directly comparable to the same model's performance under its normal $420\to384$ pipeline reported elsewhere (Table~\ref{tab:debiasing}), since it reflects a different preprocessing pipeline entirely; we do not have a basis to estimate the size of this effect from the results available. Discrimination was monitored via macro ROC-AUC and macro F1-Score, and predictive uncertainty via mean predictive entropy, computed as the average of the per-class binary entropy $-[p\log p + (1-p)\log(1-p)]$ over the 14 independent sigmoid outputs and then averaged over the test set; this is the mean of 14 binary entropies per image, not a single 14-way multiclass Shannon entropy, since the task is multi-label rather than mutually exclusive. A single severity level per corruption type, and the preprocessing mismatch just noted, are limitations of this analysis (Section~\ref{sec:limitations}); a full severity curve computed under the model's normal preprocessing pipeline would give a more reliable picture of degradation behavior. Results are summarized in Table~\ref{tab:robustness}.

\begin{table}[htbp]
\centering
\caption{Robustness of the reference model (ConvNeXt, post-intervention) under synthetic image degradation.}
\label{tab:robustness}
\begin{tabular}{lccc}
\toprule
Scenario & Macro ROC-AUC & Macro F1-Score & Mean Entropy \\
\midrule
Clean (baseline) & 0.8512 & 0.2326 & 0.1159 \\
Gaussian noise   & 0.8154 & 0.2177 & 0.1225 \\
Gaussian blur    & 0.8303 & 0.2314 & 0.1065 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig9_robustness_chart.png}
\caption{Discriminative performance (left) and mean predictive entropy (right) of the reference model under clean, Gaussian-noise, and Gaussian-blur test conditions, corresponding to the values in Table~\ref{tab:robustness}.}
\label{fig:robustness_chart}
\end{figure}

Macro ROC-AUC and macro F1-Score decreased under both evaluated perturbations relative to the (differently-preprocessed) clean baseline in this experiment. Notably, mean predictive entropy \emph{decreased} slightly under blur relative to the clean baseline, in contrast to the noise scenario, where entropy increases as expected from the addition of high-frequency stochastic perturbation. We did not test the underlying mechanism directly (e.g., via a controlled sweep of blur severity against entropy, or per-class entropy decomposition), so we offer this only as a plausible hypothesis rather than a demonstrated explanation: blur suppresses high-frequency spatial detail, which may induce a more uniform confidence response on some classes without this necessarily reflecting a clinically correct judgment. This robustness check is a single-model, single-run, two-perturbation, two-datapoint-per-perturbation-type exploratory probe, not a validated characterization of the model's behavior under corruption; we note the entropy asymmetry between noise and blur as a hypothesis worth testing with proper severity curves and repeated seeds, rather than an established, actionable property of the model, since three exploratory conditions without confidence intervals cannot support a stronger claim.

%% ============================================================
\section{Results}
\label{sec:results}

Table~\ref{tab:perclass} reports the full per-class breakdown for one specific trained instance of the post-intervention (C6) configuration at its Youden-optimal decision threshold, retained from the single-run analysis for detailed per-class inspection; per-condition variation across the three ablation seeds is characterized in aggregate in Section~\ref{sec:debiasing} rather than per class.

\begin{table}[htbp]
\centering
\caption{Per-class performance of the reference model (ConvNeXt, post-intervention), at the Youden-optimal decision threshold for each class.}
\label{tab:perclass}
\small
\begin{tabular}{lcccccc}
\toprule
Pathology & Threshold & ROC-AUC & PR-AUC & Precision & Recall & F1-Score \\
\midrule
Atelectasis         & 0.0835 & 0.8254 & 0.3539 & 0.2383 & 0.7804 & 0.3651 \\
Cardiomegaly        & 0.0318 & 0.9047 & 0.2777 & 0.1249 & 0.7563 & 0.2144 \\
Effusion            & 0.0972 & 0.8890 & 0.5396 & 0.3592 & 0.8006 & 0.4959 \\
Infiltration        & 0.1439 & 0.7147 & 0.3340 & 0.3094 & 0.5124 & 0.3858 \\
Mass                & 0.0393 & 0.8678 & 0.3805 & 0.1875 & 0.7513 & 0.3001 \\
Nodule              & 0.0398 & 0.8142 & 0.2800 & 0.1792 & 0.6473 & 0.2807 \\
Pneumonia           & 0.0134 & 0.7817 & 0.0459 & 0.0300 & 0.6492 & 0.0574 \\
Pneumothorax        & 0.0295 & 0.8971 & 0.3820 & 0.2090 & 0.8245 & 0.3334 \\
Consolidation       & 0.0392 & 0.8008 & 0.1311 & 0.1060 & 0.6731 & 0.1831 \\
Edema               & 0.0213 & 0.9072 & 0.2223 & 0.0834 & 0.8317 & 0.1517 \\
Emphysema           & 0.0181 & 0.9333 & 0.4117 & 0.1455 & 0.8289 & 0.2475 \\
Fibrosis            & 0.0080 & 0.8376 & 0.1123 & 0.0428 & 0.8106 & 0.0813 \\
Pleural Thickening  & 0.0207 & 0.8084 & 0.1252 & 0.0781 & 0.7556 & 0.1416 \\
Hernia              & 0.0028 & 0.9351 & 0.1735 & 0.0094 & 0.8261 & 0.0185 \\
\bottomrule
\end{tabular}
\end{table}

Note the very low per-class thresholds after the intervention (e.g., 0.0028 for Hernia, 0.0080 for Fibrosis), consistent with the low macro precision discussed in Section~\ref{sec:debiasing}. At this threshold, the high ROC-AUC for Hernia (0.9351) did not translate into useful positive predictive value: precision was 0.0094, reflecting the combined effect of extreme class rarity (only 23 Hernia-positive cases in the test set) and the selected sensitivity/specificity operating point.

%% ============================================================
\section{Discussion}
\label{sec:discussion}

The central methodological finding of this work is not about chest X-ray classification per se, but about evaluation practice, and it now has three parts: two that removed apparent signal under scrutiny, and one that, after initially appearing unreconciled, was internally replicated once the discrepancy was diagnosed. A single, uncontrolled before/after comparison of our explainability-motivated intervention initially suggested a clear, highly significant improvement in Grad-CAM localization against a body-silhouette proxy ($0.556\to0.601$, $p=1.4\times10^{-28}$). A controlled, seven-condition ablation with a hierarchical bootstrap (Section~\ref{sec:debiasing}) did not support this: once seed-level variability is propagated in addition to patient-level variability (albeit only partially, given the small number of seeds available; Section~\ref{sec:limitations}), no individual factor nor their combination produces a body-silhouette localization change distinguishable from ordinary training variation, and part of the original apparent effect traced to a specific, identifiable measurement artifact (a body-silhouette mask systematically excluding the patient's shoulder) discovered only through visual quality control. We then attempted to validate the same C0-vs-C6 comparison against NIH's official pathology bounding boxes (Section~\ref{sec:bbox}); an initial version of that analysis, before restricting to the held-out test split, suggested a small but hierarchically-robust pointing-game improvement, but this did not survive correcting a train/test contamination bug that had included images the model may have seen during training. Once restricted to the 131 unique test-split images with official annotations, the pointing-game effect is no longer statistically distinguishable from noise, though with limited precision for an effect of the size point-estimated (Section~\ref{sec:bbox} gives the specific sensitivity calculation); the one ground-truth localization interval that did exclude zero (a small decrease in total activation mass inside the true lesion box) is itself a single borderline result among multiple comparisons and runs counter to any improvement narrative.

Separately, and initially following the same pattern of apparent signal under scrutiny, reanalyzing the ablation's saved test predictions for calibration (Section~\ref{sec:calibration-ablation}) surfaced a large pattern: conditions using standard binary cross-entropy showed substantially lower Expected Calibration Error (roughly 29-fold lower) than conditions using Asymmetric Loss \citep{ridnik2021asymmetric}, concordant across ECE, Brier score, and NLL, with seed-to-seed variability far too small to explain it on its own. Unlike the localization effect, however, this one did not dissolve under investigation: it did not initially match a separately-computed calibration estimate for a single earlier-trained reference instance nominally following the same recipe (Section~\ref{sec:calibration}, originally reported as mean ECE 0.1015 versus $\approx0.01$ for the corresponding ablation conditions), but a targeted diagnostic (comparing checkpoint weights directly, then re-evaluating the reference instance under the ablation's own evaluation pipeline) reconciled the discrepancy and internally replicated the loss-associated pattern, without isolating the exact implementation difference in the earlier pipeline responsible for its higher estimate. We report this as an internal replication across conditions, seeds, and three calibration metrics, not as an independent validation: a useful reminder that a systematic multi-axis audit can surface consequential findings orthogonal to its original motivating question, and that not every initially-puzzling discrepancy is a sign the underlying result is wrong: some are measurement-pipeline artifacts worth finding and reconciling rather than reasons to discard the finding.

The one factor with both a confidently non-zero patient-only confidence interval and a non-trivial effect size on the body-silhouette metric was the loss function change alone (C2), and its direction (increased in-body activation) was the opposite of what a shortcut-learning-mitigation narrative would predict for the combined intervention, which instead trended negative; under the hierarchical bootstrap this interval also widens to include zero (Table~\ref{tab:localization}). Combined with the pattern observed between C1, C2, and C5, this suggested an interaction between cropping and loss function that we could not formally estimate with only seven of sixteen possible conditions run; we note it as a specific, falsifiable target for follow-up work (Section~\ref{sec:limitations}) rather than speculate beyond what the data support. It is a separate question from the calibration effect discussed above, which concerns the reliability of predicted probabilities rather than their spatial attribution.

The low operating-point precision reported for the reference single-run model (0.1468 pre-intervention, 0.1502 post-intervention; Section~\ref{sec:debiasing}) is consistent across the ablation's global metrics (Table~\ref{tab:debiasing} shows macro PR-AUC in a similarly narrow 0.271--0.279 band across all seven conditions) and is a consequence of thresholding for sensitivity/specificity balance under low class prevalence, not an effect of any studied factor. This trade-off is most visible for rare classes such as Hernia, where a test set of only 23 positive cases (Section~\ref{sec:results}) makes precision extremely sensitive to individual false positives regardless of the underlying discrimination quality (ROC-AUC 0.9351): high ranking ability, as reflected in ROC-AUC, does not by itself translate into a usable operating point under the observed prevalence and thresholding strategy. Positioned against prior work, our reference model's discrimination (macro ROC-AUC 0.852--0.855 across all ablation conditions) is competitive with recently published single-model and small-ensemble results on the same benchmark \citep{li2025chexds,yanar2025comparative}, though below the largest multi-variant ensembles \citep{fisher2025pretraining} (Table~\ref{tab:sota}; splits and preprocessing are not harmonized across these comparisons, so the table should be read as indicative positioning rather than a controlled comparison). The motivation for studying shortcut learning at all remains well founded even though our specific intervention did not show a reliable positive localization effect by any measure used here: real-world failure cases such as \citet{zech2018variable}, where hospital-specific imaging artifacts rather than pathology drove classifier decisions, and the qualitative examples in Figure~\ref{fig:shortcut} and Section~\ref{sec:xai}, confirm that shortcut learning is present in this class of models; what this work adds is a cautionary result about how difficult it is to demonstrate, and correctly measure, that a specific, low-effort mitigation reliably fixes it. Grad-CAM itself remains a post-hoc attribution method rather than a direct observation of the model's decision rule: neither the body-silhouette nor the bounding-box results are proof that the underlying decision mechanism did or did not change, and we did not run perturbation-based faithfulness checks (e.g., deletion/insertion or marker-intervention tests) that would strengthen either inference; our bounding-box check also used the ground-truth class as its Grad-CAM target, whereas the body-silhouette metric elsewhere in this work used each model's own top-1 prediction, a protocol inconsistency between our two localization analyses that we did not harmonize (Section~\ref{sec:limitations}).

On calibration overall, beyond the loss-function effect above, the corrected per-class ECE breakdown for the single reference instance (Section~\ref{sec:calibration}; mean 0.0106, range 0.0006--0.0402) indicates that even within a single well-calibrated configuration, reliability varies by pathology, with Infiltration notably higher than the rest. This qualification matters in practice because, per \citet{rajaraman2022calibration}, calibration gains in imbalanced medical-imaging classifiers tend to be concentrated at a fixed default threshold (0.5) and to shrink once thresholds are already tuned via ROC/PR curves, as they are throughout this work. We did not apply a post-hoc calibration method (e.g., temperature or Platt scaling) to any condition; all calibration numbers reported here characterize native, uncalibrated model outputs.

On robustness, ROC-AUC and F1-score decreased under both noise and blur relative to the clean condition in this experiment (Section~\ref{sec:robustness}; note the preprocessing caveat discussed there), and the counter-intuitive entropy \emph{decrease} under blur is worth flagging as a hypothesis for future work, though not as an established property of the model given this is a single-run, three-condition, no-confidence-interval exploratory probe: a naive entropy-based deferral rule (routing low-confidence cases to a human reader) would, if this pattern held up under proper testing, become \emph{less} likely to flag blurred images for human review precisely when review might be most needed, since blur appears to lower rather than raise the model's apparent uncertainty here. We make no claim about deployment readiness under real (non-synthetic) acquisition variability, which two synthetic corruptions tested once on a single internal dataset cannot establish.

\subsection{Limitations}
\label{sec:limitations}
This work has several limitations. First, although the multi-condition ablation and hierarchical bootstrap (Section~\ref{sec:debiasing}) address the single-run and pseudoreplication confounds of the original intervention comparison, the body-silhouette localization metric was evaluated under a single, shared preprocessing choice ($420\to384$ resize-and-crop) for all seven conditions, so conditions without crop as a native factor are evaluated on a crop none of them were trained with; the bounding-box validation (Section~\ref{sec:bbox}) uses the same shared preprocessing for the same reason. Second, the corrected body-silhouette mask still uses a fixed morphological closing radius (20\,px) chosen by visual inspection of a handful of failure cases rather than validated against ground-truth anatomy. Third, this ablation is not a full factorial design: with seven of sixteen possible $2^4$ conditions run, the crop-loss pattern observed between C1, C2, and C5 is only suggestive of an interaction, not a formally estimated one; a complete $2^4$ design, ideally with five or more seeds, would be needed to characterize this with confidence. Fourth, the bounding-box validation (Section~\ref{sec:bbox}), once correctly restricted to the test split, has only 131 unique annotated images; we compute and report a specific minimum-detectable-effect estimate (Section~\ref{sec:bbox}) rather than only asserting the sample is small, but a larger annotated validation set (external or from a different NIH release) would still be needed to resolve whether the point-estimated effect is real. Fifth, the bounding-box validation was run only for C0 and C6, not the intermediate conditions, so no effect (had one survived) could have been attributed to a specific factor; it also used the ground-truth class as the Grad-CAM target rather than the model's own top-1 prediction used for the body-silhouette metric elsewhere in this work, an inconsistency between the two localization protocols that should be harmonized in follow-up work, and we did not correct for testing multiple localization endpoints (pointing-game and energy-in-box) without a pre-specified primary one. Sixth, the calibration discrepancy that this paper initially reported as unresolved (an earlier single-instance ECE estimate of 0.1015 versus $\approx0.01$ for the corresponding ablation conditions) has since been traced to the earlier evaluation pipeline used for that single-instance analysis, confirmed by re-evaluating the same checkpoint under the ablation's pipeline (Section~\ref{sec:calibration}); we did not fully characterize which specific aspect of the earlier pipeline (e.g., preprocessing details not otherwise documented) produced the discrepancy, only that the model itself was not its source, and a complete diff of the two pipeline implementations would still be worthwhile for full reproducibility. Seventh, Grad-CAM is a post-hoc attribution method, not a direct observation of the model's decision mechanism, and we did not conduct perturbation-based faithfulness checks. Eighth, the robustness evaluation used a single noise level and a single blur kernel size rather than a severity curve, with no confidence intervals and only one trained instance, and did not include acquisition-relevant corruptions beyond Gaussian noise and blur. Ninth, the NIH ChestX-ray14 labels were extracted automatically via NLP with a reported precision above 90\%, so a nonzero fraction of labels are noisy by construction, and all results are reported on a single-institution dataset with no external (cross-hospital) validation. Tenth, no subgroup analysis by sex, age, or view position was performed. Eleventh, the choice of ConvNeXt as the reference backbone (Section~\ref{sec:architectures}) was based on an unblinded, non-predefined visual judgment of Grad-CAM map quality by the authors. Twelfth, with only three seeds, all hierarchical-bootstrap intervals in this work (27 possible seed resamples) partially, not fully, account for between-run variability, and inference from three training runs remains statistically unstable; five or more seeds, particularly for C0 and C6, would substantially strengthen every statistical claim in this paper. Finally, our shortcut-learning checks remain aggregate spatial measures rather than attribute-specific tests (e.g., for hospital source or patient demographics, as in \citet{brown2023detecting}). Given these limitations, we frame this work as an exploratory, single-dataset audit with a predominantly null result on its central localization question across two different measurement approaches, alongside a large and now-reconciled calibration finding, rather than as a validated claim of clinical readiness, trustworthiness, or a working bias-mitigation method.

%% ============================================================
\section{Conclusion}
\label{sec:conclusion}
This work presented a multi-axis audit (discriminative performance across eight backbone/pretraining configurations and two ensembles, an explainability-motivated intervention, probability calibration, and robustness to synthetic degradation) applied to ConvNeXt-based multi-label classification of 14 thoracic pathologies on NIH ChestX-ray14. Discrimination is stable and competitive with recently published single-model and small-ensemble results (macro ROC-AUC 0.852--0.855 across all seven ablation conditions). The central finding concerns the explainability-motivated intervention, and unfolds across three layers of increasingly careful scrutiny. A single, uncontrolled before/after comparison initially indicated a strong, highly significant improvement in Grad-CAM localization against a body-silhouette proxy; a controlled, seven-condition ablation with a hierarchical bootstrap found no such effect once a shoulder-exclusion mask artifact was corrected and seed-level variance was propagated, albeit only partially given the small number of seeds available. Validating the same comparison against NIH's official pathology bounding boxes, restricted to the 131 unique test-split images with official annotations after correcting an initial train/test contamination bug in that analysis, likewise found no statistically robust improvement in pointing-game accuracy, with limited precision given the sample size; the one ground-truth localization interval that did exclude zero (a small decrease in activation mass inside the true lesion box) is a single borderline, uncorrected-for-multiplicity result that runs counter to an improvement narrative. Independently, reanalysis of the same experimental infrastructure for calibration surfaced a large pattern linking loss function to Expected Calibration Error, concordant across three metrics; unlike the localization effect, this one did not dissolve under scrutiny but was instead internally replicated once an initial discrepancy with a separately-produced single-instance estimate was reconciled, via a released diagnostic protocol, against an earlier evaluation pipeline rather than the model itself; we report the corrected, reconciled result (a roughly 29-fold difference between loss functions) as an internally replicated pattern of this paper, not an independently validated one. We report this full trajectory, and not any single number, as the primary contribution of this paper: it demonstrates concretely how a plausible, single-run debiasing narrative can fail to survive proper statistical and measurement controls at multiple successive levels of scrutiny, how limited sample sizes and multiple comparisons can produce borderline results that should not be over-interpreted in either direction, and how a systematic multi-axis audit can both surface and, with appropriate diagnostic follow-through, reconcile and internally replicate a large finding orthogonal to its original question even as its central hypothesis fails to hold up. We do not claim to have established clinical trustworthiness, screening readiness, or a working debiasing method for this model. Future work should prioritize external, cross-institution validation; a larger ground-truth-annotated localization sample with adequate statistical power; a full diff of the two calibration evaluation pipelines to document the exact source of the now-resolved discrepancy; at least two additional training seeds for C0 and C6 specifically; a complete $2^4$ factorial design with five or more seeds; and attribute-specific shortcut testing to complement the aggregate checks presented here.

%% ============================================================
\section*{CRediT authorship contribution statement}
\textbf{Luca Migliaccio}: Methodology, Software, Investigation, Data curation, Writing, original draft. \textbf{Antonio Esposito}: Conceptualization, Supervision, Writing, review \& editing.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Ethical considerations}
This study uses the publicly released NIH ChestX-ray14 dataset under its original terms of use; no new patient data were collected. This is a secondary analysis of de-identified public data; we have not documented a formal institutional review exemption for it. We did not assess subgroup performance by sex, age, or view position (Section~\ref{sec:limitations}), so no fairness claims are made or should be inferred. This work has not been checked against a formal reporting checklist such as CLAIM or TRIPOD-AI.

\section*{Data availability}
The NIH ChestX-ray14 dataset is publicly available. The training, evaluation, calibration, robustness, and explainability code developed for this study, including the multi-condition ablation, patient-clustered and hierarchical bootstrap analysis scripts, and the ECE-discrepancy diagnostic script that traced and resolved the calibration-pipeline discrepancy discussed in Section~\ref{sec:calibration-ablation}, is available at \url{https://github.com/whitewizard85/xRayProject}, with local file paths parameterized for portability where applicable. Per-condition, per-seed global metrics, per-image body-silhouette localization measurements (Table~\ref{tab:debiasing}, Table~\ref{tab:localization}), the list of 131 test-split images with official bounding-box annotations and per-annotation pointing-game/energy-in-box results (Table~\ref{tab:bbox}), and per-image, per-class predicted probabilities underlying the calibration analysis (Table~\ref{tab:ece}, Table~\ref{tab:calib-ablation}, including the single-seed patient-clustered ECE confidence intervals omitted from the printed table) are released as CSV files in the \texttt{paper\_reproducibility/results/} directory of the repository. Checkpoint identifiers (content hashes) for every trained model instance, including the reference single-instance checkpoint and its diagnostic comparison against the ablation's C6 checkpoints, are included to support exact reproducibility checks. Trained model checkpoints themselves are not tracked in the repository and will be hosted separately (e.g., Zenodo) if released.

\section*{Acknowledgements}
The authors thank the Department of Engineering, Universit\`a degli Studi della Campania ``Luigi Vanvitelli,'' for providing the computational infrastructure used in this study.

%% ============================================================
\bibliographystyle{elsarticle-num-names}
\bibliography{references}

\end{document}
