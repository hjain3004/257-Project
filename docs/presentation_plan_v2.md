# Sepsis Early Warning Pipeline: From Failures to a Calibrated Multi-Modal Ensemble

**Gamma AI Prompting Instructions:** 
*This document contains the exact text, flow, and structural layout for a technical Data Science presentation. Do not dilute the technical terminology (e.g., AUPRC, Isotonic Regression, GRU-D). Keep the slides heavily focused on data visualization rather than text. Where an image is specified, leave a large visual placeholder for the user to drop the PNG.*

---

## Slide 1: Title Slide
- **Title:** Early Warning System for ICU Sepsis: A Multi-Modal Calibrated Ensemble
- **Subtitle:** CMPE 257 — Group 7 (Nitish, Himanshu, Krishna, Lavya)
- **Visual:** [No specific figure, keep clean and professional]

---

## Slide 2: The Clinical Motivation
- **Headline:** Every hour of delayed sepsis treatment increases mortality by 4–7%.
- **Bullet points:**
  - Sepsis causes ~270,000 deaths annually in the US.
  - ICU mortality reaches ~30% if untreated early.
  - The goal is an actionable 6-hour early warning window prior to clinical recognition.
- **Visual:** [Abstract medical/ICU data background]

---

## Slide 3: The Dataset & The Machine Learning Problem
- **Headline:** PhysioNet 2019 Challenge Dataset
- **Bullet points:**
  - **Scale:** 40,336 ICU patients, 1.55 Million hourly rows, 40 clinical features (Vitals + Labs) across 2 hospitals.
  - **The Target:** Binary classification to predict `SepsisLabel == 1` exactly 6 hours before clinical onset.
  - **The Challenge:** Extreme class imbalance. Only 2.2% of the training dataset rows represent positive sepsis cases. Plain accuracy metrics are mathematically meaningless here.
- **Image Placeholder:** `dataset_overview.png`
- **Location in Repo:** `development/figures/dataset_overview.png`

---

## Slide 4: Phase 1 — Where We Went Wrong (Initial Failures)
- **Headline:** Initial naive modeling exposed critical challenges in medical data science.
- **Bullet points:**
  - **Leakage via Naive Splitting:** A standard row-level random split leaked temporal data (e.g., Patient A's Hour 3 in train, Hour 5 in test).
  - **Destroyed Signals via Mean Imputation:** Filling 85% missing lab values with the median destroyed "informative missingness" (the clinical signal of a doctor actually *ordering* a lab test).
  - **Imbalance Mismanagement:** Models trained on the 45:1 imbalance outputted "No Sepsis" for everyone, achieving 97% accuracy but 0% recall. Undersampling threw away 96% of valuable data.
- **Visual:** [Iconography showing failure/iteration loop]

---

## Slide 5: Handling Missingness & Feature Engineering
- **Headline:** Treating missing data as a physiological signal.
- **Bullet points:**
  - **Informative Missingness:** Labs are missing >80% of the time, but the presence of a measurement is highly correlated with clinical suspicion of deterioration.
  - **Tabular Imputation:** We shifted from mean-fill to patient-level forward-fill, combined with 6h rolling statistics and explicit boolean missingness masks.
  - **Biological Verification:** Aligning patient trajectories 24 hours prior to onset confirmed textbook Systemic Inflammatory Response Syndrome (SIRS) signatures.
- **Image Placeholder:** `vital_trajectories.png`
- **Location in Repo:** `development/figures/vital_trajectories.png`

---

## Slide 6: The Corrected Pipeline Architecture
- **Headline:** A robust, leak-free, multi-modal pipeline.
- **Bullet points:**
  - **Strict Patient-Level Splits:** Guaranteed zero patient overlap between train/val/test splits to prevent temporal leakage.
  - **Multi-Modal Inductive Biases:** Instead of one model, we routed the data into three specialized pipelines to capture cross-sectional, sequential, and visual representations.
  - **Evaluation Overhaul:** Shifted entirely from Accuracy to Area Under the Precision-Recall Curve (AUPRC), the gold standard for highly imbalanced medical datasets.
- **Visual:** [Create a flowchart in Gamma: Raw Data → Split into (1. Tabular, 2. Sequences, 3. Images) → Meta-Stacker]

---

## Slide 7: Modality 1 — RF & XGBoost (The Tabular View & Tuning)
- **Headline:** Rigorous Hyperparameter Tuning to Combat Overfitting.
- **Bullet points:**
  - **Feature Engineering:** Expanded 40 raw features to 198 dimensions (deltas, slopes, baseline deviations, qSOFA, NEWS).
  - **The Overfitting Problem:** Out-of-the-box Random Forest and XGBoost completely overfit to the severe class imbalance, capturing noise instead of signal.
  - **Hyperparameter Optimization:** We ran extensive hyperparameter searches to strictly regularize the models.
    - *XGBoost Tuning:* Restricted `max_depth` to 4 (shallow trees), lowered `learning_rate` to 0.03, and scaled to 1,049 estimators.
    - *Random Forest Tuning:* Tuned `min_samples_split` and `max_features` to ensure robust tree generalization.
  - **Top Drivers:** Post-tuning, physiological markers like Lactate, Heart Rate, and Age dominated the gain importance.
- **Image Placeholder:** `xgb_top20.png`
- **Location in Repo:** `development/figures/xgb_top20.png`

---

## Slide 8: Modality 2 — GRU-D (The Temporal Sequence View)
- **Headline:** Natively handling irregular time-gaps via decay mechanisms.
- **Bullet points:**
  - **The Problem:** Standard RNNs assume uniform time steps, which fails in the ICU.
  - **The Solution:** GRU-D learns explicit decay parameters ($\gamma$) for hidden states and inputs based on the `hours_since_measured`. 
  - **Clinical Translation:** A heart rate from 10 minutes ago is trusted; a lactate from 18 hours ago decays toward the clinical baseline.
- **Visual:** [Use the GRU-D architecture diagram from Che et al. 2016 paper if possible, or a neural network graphic]

---

## Slide 9: Modality 3 — GAF-CNN (The Computer Vision View)
- **Headline:** Encoding physiological time-series as image textures.
- **Bullet points:**
  - **Gramian Angular Fields (GAF):** Translated 24-hour sliding windows of 1D vitals into 2D polar coordinate matrices.
  - **CNN Extraction:** Leveraged Convolutional Neural Networks to detect visual "textures" corresponding to sepsis onset.
  - **Ensemble Diversity:** Provided a completely distinct mathematical representation of patient deterioration.
- **Image Placeholder:** [Any example GAF image grid generated from `sepsis_gaf.ipynb`]
- **Location in Repo:** `development/figures/` (User must provide GAF examples)

---

## Slide 10: Meta-Stacking & Probability Calibration
- **Headline:** Combining diverse views and tuning for clinical reality.
- **Bullet points:**
  - **Isotonic Regression:** Raw model scores are mathematically uncalibrated. We applied Isotonic Regression to map outputs to true clinical probabilities.
  - **Logistic Regression Stacker:** A meta-learner combining calibrated XGBoost and GRU-D probabilities.
  - **Stacker Weights:** Nearly balanced (8.0 vs 9.3), proving that both tabular and sequence models contribute unique, independent predictive power.
- **Image Placeholder:** `ensemble_diagram.png`
- **Location in Repo:** `development/figures/ensemble_diagram.png`

---

## Slide 11: Headline Metrics & Baseline Lift
- **Headline:** Delivering a 5x lift over the baseline probability.
- **Bullet points:**
  - **Base Rate Reality:** With a 2.2% positive prevalence, a random, no-skill classifier achieves an AUPRC of exactly 0.022.
  - **Ensemble Performance:** Our Meta-Stacker achieved an AUPRC of 0.1097.
  - **The Impact:** This represents a 5x mathematical improvement over random chance, pushing the AUROC past the 0.80 "good" threshold for clinical machine learning.
- **Image Placeholder:** `headline_metrics.png`
- **Location in Repo:** `development/figures/headline_metrics.png`

---

## Slide 12: ROC and Precision-Recall Curves
- **Headline:** The Ensemble strictly dominates the base models.
- **Bullet points:**
  - **ROC Curve:** All models perform near the 0.81 AUC threshold, indicating strong general discriminative ability.
  - **PR Curve:** The critical metric for imbalanced data. The Ensemble perfectly smooths the recall-precision trade-off, maximizing the capture of True Positives while resisting False Positives.
- **Image Placeholder:** `curves_overlay.png`
- **Location in Repo:** `development/figures/curves_overlay.png`

---

## Slide 13: Early Warning Lead Time (The True Value)
- **Headline:** Catching Sepsis 8 hours before doctors realize it.
- **Bullet points:**
  - **Median Lead Time:** The model successfully flagged patient deterioration a median of 8.0 hours in advance of the official Sepsis-3 diagnosis.
  - **Early Warning Rate:** 67.4% of all caught sepsis cases were flagged at least 6 hours ahead of time.
  - **Clinical Rule Baseline:** Significantly outperformed existing hospital heuristics (qSOFA and NEWS scores).
- **Image Placeholder:** `lead_time_histogram.png`
- **Location in Repo:** `development/output/figures/lead_time_histogram.png`

---

## Slide 14: Patient Case Study — A True Positive
- **Headline:** Visualizing the 8-hour head start on a real patient timeline.
- **Bullet points:**
  - This patient trajectory highlights the exact moment the model probability crossed the clinical threshold.
  - Note the clear divergence in physiological markers (Heart Rate, MAP) aligning with the model's rising confidence.
  - The alert fired hours before the red "onset" line, granting clinicians crucial time to administer antibiotics.
- **Image Placeholder:** `tp_1_pid18719.png` (or any `tp_*.png`)
- **Location in Repo:** `development/output/figures/case_studies/tp_1_pid18719.png`

---

## Slide 15: Cross-Hospital Generalization
- **Headline:** Proving robust generalization on entirely unseen external data.
- **Bullet points:**
  - **The Test:** Validating a model on the same hospital it was trained on often hides overfitting. We held out Hospital B entirely.
  - **The Result:** The GRU-D model maintained a 0.77 AUROC on Hospital B, experiencing only a minor 1.5-point drop from internal testing.
  - **Conclusion:** The learned physiological signatures represent fundamental sepsis pathophysiology, not hospital-specific protocols.
- **Image Placeholder:** `cross_hospital.png`
- **Location in Repo:** `development/figures/cross_hospital.png`

---

## Slide 16: Literature Comparison & Future Work
- **Headline:** Matching state-of-the-art PhysioNet Challenge benchmarks.
- **Bullet points:**
  - **Benchmark Validation:** The top-winning teams of the official PhysioNet 2019 challenge achieved AUROCs between 0.78 and 0.83. Our 0.81 Ensemble sits securely in the top-tier range.
  - **Future Work:**
    - Subgroup fairness analysis (Age, Gender, ICU type).
    - Implementing temporal cross-validation.
    - Live shadow deployment for false-alarm fatigue testing.
- **Image Placeholder:** `lit_comparison.png`
- **Location in Repo:** `development/figures/lit_comparison.png`
