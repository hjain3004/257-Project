

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import os
import pickle


PARQUET_PATH = os.path.join(os.path.dirname(__file__), "combined_data.parquet")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed_data")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Feature groups
VITAL_SIGNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
LAB_VALUES = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]
CLINICAL_FEATURES = VITAL_SIGNS + LAB_VALUES  # 33 features (EtCO2 excluded)


def generate_pipeline_diagram(output_dir):
    """Create a visual diagram of the preprocessing pipeline."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')

    steps = [
        (1, "20,336 PSV\nFiles Loaded", '#E3F2FD'),
        (3, "Drop EtCO2\n(100% missing)", '#FFF3E0'),
        (5, "Missingness\nIndicators\n(33 features)", '#E8F5E9'),
        (7, "Forward Fill\n(per patient)", '#F3E5F5'),
        (9, "Median\nImputation", '#FFF9C4'),
        (11, "Feature\nEngineering\n(3 new)", '#E0F7FA'),
        (13, "Standardize\n& Split\n80/20", '#FCE4EC'),
    ]
    for x, text, color in steps:
        rect = plt.Rectangle((x - 0.7, 0.5), 1.4, 2, facecolor=color,
                              edgecolor='black', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 1.5, text, ha='center', va='center', fontsize=8,
                fontweight='bold', zorder=3)

    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i + 1][0] - 0.7, 1.5),
                    xytext=(steps[i][0] + 0.7, 1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preprocessing_pipeline.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: preprocessing_pipeline.png")


def preprocess(combined):
    """Run the full preprocessing pipeline."""

    # ----------------------------------------------------------
    # Step 1: Drop EtCO2 (100% missing)
    # ----------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Drop EtCO2 (100% missing)")
    print("=" * 60)
    combined.drop(columns=['EtCO2'], inplace=True, errors='ignore')
    print(f"Shape after dropping EtCO2: {combined.shape}")

    # ----------------------------------------------------------
    # Step 2: Create missingness indicator features
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Create Missingness Indicator Features")
    print("=" * 60)
    for feat in CLINICAL_FEATURES:
        combined[f'{feat}_missing'] = combined[feat].isnull().astype(int)
    print(f"Created {len(CLINICAL_FEATURES)} missingness indicator features")
    print(f"Shape: {combined.shape}")

    # ----------------------------------------------------------
    # Step 3: Forward fill within each patient
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Forward Fill (Last Observation Carried Forward)")
    print("=" * 60)
    combined = combined.sort_values(['PatientID', 'ICULOS']).reset_index(drop=True)
    clinical_and_demo = CLINICAL_FEATURES + ['Unit1', 'Unit2']
    combined[clinical_and_demo] = combined.groupby('PatientID')[clinical_and_demo].ffill()

    missing_after = combined[CLINICAL_FEATURES].isnull().mean() * 100
    print(f"After forward fill:")
    print(f"  Vital signs avg missing: {missing_after[VITAL_SIGNS].mean():.1f}%")
    print(f"  Lab values avg missing:  {missing_after[LAB_VALUES].mean():.1f}%")

    # ----------------------------------------------------------
    # Step 4: Median imputation for remaining NaNs
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Median Imputation for Remaining NaNs")
    print("=" * 60)
    medians = combined[CLINICAL_FEATURES].median()
    combined[CLINICAL_FEATURES] = combined[CLINICAL_FEATURES].fillna(medians)
    combined['Unit1'] = combined['Unit1'].fillna(0)
    combined['Unit2'] = combined['Unit2'].fillna(0)
    combined['HospAdmTime'] = combined['HospAdmTime'].fillna(combined['HospAdmTime'].median())
    print(f"Remaining NaNs: {combined.isnull().sum().sum()}")

    # ----------------------------------------------------------
    # Step 5: Feature engineering
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Feature Engineering")
    print("=" * 60)

    # Shock Index = HR / SBP (hemodynamic instability marker)
    combined['ShockIndex'] = combined['HR'] / combined['SBP'].replace(0, np.nan)
    combined['ShockIndex'] = combined['ShockIndex'].fillna(combined['ShockIndex'].median())

    # MAP-to-HR ratio (perfusion pressure vs cardiac demand)
    combined['MAP_HR_ratio'] = combined['MAP'] / combined['HR'].replace(0, np.nan)
    combined['MAP_HR_ratio'] = combined['MAP_HR_ratio'].fillna(combined['MAP_HR_ratio'].median())

    # BUN/Creatinine ratio (renal function indicator)
    combined['BUN_Creat_ratio'] = combined['BUN'] / combined['Creatinine'].replace(0, np.nan)
    combined['BUN_Creat_ratio'] = combined['BUN_Creat_ratio'].fillna(
        combined['BUN_Creat_ratio'].median())

    print("Created: ShockIndex, MAP_HR_ratio, BUN_Creat_ratio")
    print(f"Shape: {combined.shape}")

    # ----------------------------------------------------------
    # Step 6: Patient-level train/test split (80/20)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Patient-Level Train/Test Split (80/20)")
    print("=" * 60)
    patient_ids = combined['PatientID'].unique()
    patient_labels = combined.groupby('PatientID')['SepsisLabel'].max()

    patient_df = pd.DataFrame({'PatientID': patient_ids})
    patient_df['label'] = patient_df['PatientID'].map(patient_labels)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(patient_df, patient_df['label'],
                                         groups=patient_df['PatientID']))

    train_patients = set(patient_df.iloc[train_idx]['PatientID'])
    test_patients = set(patient_df.iloc[test_idx]['PatientID'])

    train_data = combined[combined['PatientID'].isin(train_patients)].copy()
    test_data = combined[combined['PatientID'].isin(test_patients)].copy()

    print(f"Train: {len(train_patients)} patients, {len(train_data):,} rows")
    print(f"Test:  {len(test_patients)} patients, {len(test_data):,} rows")
    print(f"Train sepsis rate: {train_data.groupby('PatientID')['SepsisLabel'].max().mean() * 100:.2f}%")
    print(f"Test sepsis rate:  {test_data.groupby('PatientID')['SepsisLabel'].max().mean() * 100:.2f}%")

    # ----------------------------------------------------------
    # Step 7: Standardization (fit on train, transform both)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Standardization")
    print("=" * 60)
    scale_features = CLINICAL_FEATURES + ['ShockIndex', 'MAP_HR_ratio',
                                           'BUN_Creat_ratio', 'HospAdmTime']
    scaler = StandardScaler()
    train_data[scale_features] = scaler.fit_transform(train_data[scale_features])
    test_data[scale_features] = scaler.transform(test_data[scale_features])
    print(f"Standardized {len(scale_features)} continuous features")

    return train_data, test_data, scaler, medians


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Load combined data
    print("Loading combined data...")
    combined = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {combined.shape}")

    # Run preprocessing
    train_data, test_data, scaler, medians = preprocess(combined)

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    train_data.to_parquet(os.path.join(PROCESSED_DIR, "train_data.parquet"), index=False)
    test_data.to_parquet(os.path.join(PROCESSED_DIR, "test_data.parquet"), index=False)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, "medians.pkl"), 'wb') as f:
        pickle.dump(medians, f)

    print(f"Saved train_data.parquet ({len(train_data):,} rows)")
    print(f"Saved test_data.parquet ({len(test_data):,} rows)")
    print(f"Saved scaler.pkl and medians.pkl")

    # Generate pipeline diagram
    generate_pipeline_diagram(FIGURE_DIR)

    # Final summary
    all_feat_cols = [c for c in train_data.columns
                     if c not in ['PatientID', 'SepsisLabel', 'ICULOS']]
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total features: {len(all_feat_cols)}")
    print(f"  Clinical (scaled): {len(CLINICAL_FEATURES)}")
    print(f"  Missingness indicators: {len(CLINICAL_FEATURES)}")
    print(f"  Engineered: 3 (ShockIndex, MAP_HR_ratio, BUN_Creat_ratio)")
    print(f"  Demographics: 5 (Age, Gender, Unit1, Unit2, HospAdmTime)")
