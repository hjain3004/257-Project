

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "training")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "combined_data.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature groups
VITAL_SIGNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
LAB_VALUES = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]
DEMOGRAPHICS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']



def load_all_patients(data_dir):
    """Load all .psv files into a single DataFrame with PatientID column."""
    psv_files = sorted(glob.glob(os.path.join(data_dir, "p*.psv")))
    print(f"Found {len(psv_files)} patient files")

    dfs = []
    for f in psv_files:
        pid = os.path.basename(f).replace('.psv', '')
        df = pd.read_csv(f, sep='|')
        df['PatientID'] = pid
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    print(f"Total hourly records: {len(combined):,}")
    print(f"Total patients: {combined['PatientID'].nunique():,}")
    return combined



def print_dataset_summary(combined):
    """Print comprehensive summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {combined.shape}")
    print(f"Total rows (hourly records): {len(combined):,}")
    print(f"Total patients: {combined['PatientID'].nunique():,}")
    print(f"Total features (columns): {combined.shape[1]}")

    print("\n--- Column Types ---")
    print(combined.dtypes.to_string())

    # ICU stay lengths
    stay_lengths = combined.groupby('PatientID')['ICULOS'].max()
    print(f"\n--- ICU Stay Lengths ---")
    print(f"Min: {stay_lengths.min()}h, Max: {stay_lengths.max()}h, "
          f"Median: {stay_lengths.median():.0f}h, Mean: {stay_lengths.mean():.1f}h")

    # Class balance (patient-level)
    sepsis_per_patient = combined.groupby('PatientID')['SepsisLabel'].max()
    pos = (sepsis_per_patient == 1).sum()
    neg = (sepsis_per_patient == 0).sum()
    print(f"\n--- Class Balance (Patient-Level) ---")
    print(f"Sepsis-positive: {pos} ({pos / len(sepsis_per_patient) * 100:.2f}%)")
    print(f"Sepsis-negative: {neg} ({neg / len(sepsis_per_patient) * 100:.2f}%)")

    # Class balance (row-level)
    vc = combined['SepsisLabel'].value_counts().sort_index()
    print(f"\n--- Class Balance (Row-Level) ---")
    print(f"SepsisLabel=0: {vc[0]:,} ({vc[0] / len(combined) * 100:.2f}%)")
    print(f"SepsisLabel=1: {vc[1]:,} ({vc[1] / len(combined) * 100:.2f}%)")

    # Missing values
    print(f"\n--- Missing Values by Feature Group ---")
    print(f"Vital signs avg missing: {combined[VITAL_SIGNS].isnull().mean().mean() * 100:.1f}%")
    print(f"Lab values avg missing:  {combined[LAB_VALUES].isnull().mean().mean() * 100:.1f}%")
    print(f"Demographics avg missing: {combined[DEMOGRAPHICS].isnull().mean().mean() * 100:.1f}%")

    # Per-feature missing
    print(f"\n--- Missing Values Per Feature ---")
    feature_cols = [c for c in combined.columns if c != 'PatientID']
    missing_pct = (combined[feature_cols].isnull().mean() * 100).sort_values(ascending=False)
    print(missing_pct.round(2).to_string())

    # Demographics
    demo = combined.groupby('PatientID').first()
    print(f"\n--- Demographics ---")
    print(f"Age: mean={demo['Age'].mean():.1f}, std={demo['Age'].std():.1f}, "
          f"min={demo['Age'].min()}, max={demo['Age'].max()}")
    print(f"Gender: Female(0)={(demo['Gender'] == 0).sum()}, Male(1)={(demo['Gender'] == 1).sum()}")

    # Vital signs summary
    print(f"\n--- Vital Signs Summary ---")
    print(combined[VITAL_SIGNS].describe().round(2).to_string())



def plot_missing_values_bar(combined, output_dir):
    """Bar chart of missing value percentage per feature."""
    all_feats = VITAL_SIGNS + LAB_VALUES
    missing_pct = [(combined[f].isnull().mean() * 100) for f in all_feats]
    colors = ['#2196F3'] * len(VITAL_SIGNS) + ['#FF9800'] * len(LAB_VALUES)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(all_feats)), missing_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(all_feats)))
    ax.set_xticklabels(all_feats, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Missing %')
    ax.set_title('Missing Value Percentage by Feature', fontsize=13, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc='#2196F3', label='Vital Signs'),
        plt.Rectangle((0, 0), 1, 1, fc='#FF9800', label='Lab Values'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='50% threshold')
    ], labels=['Vital Signs', 'Lab Values', '50% threshold'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: missing_values_bar.png")


def plot_missing_values_heatmap(combined, output_dir):
    """Heatmap of missing value percentages."""
    all_feats = VITAL_SIGNS + LAB_VALUES + DEMOGRAPHICS
    missing_pct = (combined[all_feats].isnull().mean() * 100).values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(missing_pct, annot=False, cmap='YlOrRd', xticklabels=all_feats,
                yticklabels=[''], ax=ax, cbar_kws={'label': 'Missing %'}, vmin=0, vmax=100)
    ax.set_title('Missing Value Percentage Across All Features', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: missing_values_heatmap.png")


def plot_class_distribution(combined, output_dir):
    """Patient-level and row-level class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Patient-level
    sepsis_patient = combined.groupby('PatientID')['SepsisLabel'].max()
    counts = sepsis_patient.value_counts().sort_index()
    bars = axes[0].bar(['No Sepsis (0)', 'Sepsis (1)'], counts.values,
                       color=['#2196F3', '#f44336'], edgecolor='black')
    axes[0].set_title('Patient-Level Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Patients')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f'{val:,}\n({val / counts.sum() * 100:.1f}%)', ha='center', fontsize=10)

    # Row-level
    row_counts = combined['SepsisLabel'].value_counts().sort_index()
    bars2 = axes[1].bar(['No Sepsis (0)', 'Sepsis (1)'], row_counts.values,
                        color=['#2196F3', '#f44336'], edgecolor='black')
    axes[1].set_title('Row-Level (Hourly) Class Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Hourly Records')
    for bar, val in zip(bars2, row_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                     f'{val:,}\n({val / row_counts.sum() * 100:.1f}%)', ha='center', fontsize=10)

    plt.suptitle('Severe Class Imbalance in the PhysioNet 2019 Dataset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: class_distribution.png")


def plot_vital_signs_distributions(combined, output_dir):
    """Vital sign distributions stratified by sepsis status."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feat in enumerate(VITAL_SIGNS):
        ax = axes[i]
        data_0 = combined.loc[combined['SepsisLabel'] == 0, feat].dropna()
        data_1 = combined.loc[combined['SepsisLabel'] == 1, feat].dropna()
        if len(data_0) > 50000:
            data_0 = data_0.sample(50000, random_state=42)
        if len(data_1) > 50000:
            data_1 = data_1.sample(50000, random_state=42)
        ax.hist(data_0, bins=50, alpha=0.6, color='#2196F3', label='No Sepsis', density=True)
        ax.hist(data_1, bins=50, alpha=0.6, color='#f44336', label='Sepsis', density=True)
        ax.set_title(feat, fontweight='bold')
        ax.legend(fontsize=7)

    axes[7].set_visible(False)  # EtCO2 is 100% missing
    fig.suptitle('Vital Sign Distributions: Sepsis vs Non-Sepsis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vital_signs_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: vital_signs_distributions.png")


def plot_icu_stay_distribution(combined, output_dir):
    """ICU length of stay distribution by sepsis status."""
    fig, ax = plt.subplots(figsize=(10, 5))
    stay_lens = combined.groupby('PatientID')['ICULOS'].max()
    sepsis_flag = combined.groupby('PatientID')['SepsisLabel'].max()

    stay_0 = stay_lens[sepsis_flag == 0]
    stay_1 = stay_lens[sepsis_flag == 1]
    ax.hist(stay_0, bins=80, alpha=0.6, color='#2196F3',
            label=f'No Sepsis (n={len(stay_0):,})', density=True)
    ax.hist(stay_1, bins=80, alpha=0.6, color='#f44336',
            label=f'Sepsis (n={len(stay_1):,})', density=True)
    ax.set_xlabel('ICU Length of Stay (hours)')
    ax.set_ylabel('Density')
    ax.set_title('ICU Length of Stay Distribution by Sepsis Status', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "icu_stay_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: icu_stay_distribution.png")


def plot_demographics(combined, output_dir):
    """Demographics analysis: age, gender, sepsis rate by age."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    demo = combined.groupby('PatientID').first()
    demo['SepsisLabel'] = combined.groupby('PatientID')['SepsisLabel'].max()

    # Age distribution
    axes[0].hist(demo.loc[demo['SepsisLabel'] == 0, 'Age'], bins=30, alpha=0.6,
                 color='#2196F3', label='No Sepsis', density=True)
    axes[0].hist(demo.loc[demo['SepsisLabel'] == 1, 'Age'], bins=30, alpha=0.6,
                 color='#f44336', label='Sepsis', density=True)
    axes[0].set_title('Age Distribution', fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].legend()

    # Gender
    gender_sepsis = demo.groupby(['Gender', 'SepsisLabel']).size().unstack(fill_value=0)
    gender_sepsis.plot(kind='bar', ax=axes[1], color=['#2196F3', '#f44336'], edgecolor='black')
    axes[1].set_title('Gender vs Sepsis', fontweight='bold')
    axes[1].set_xticklabels(['Female (0)', 'Male (1)'], rotation=0)
    axes[1].legend(['No Sepsis', 'Sepsis'])

    # Sepsis rate by age bins
    demo['AgeBin'] = pd.cut(demo['Age'], bins=[0, 30, 40, 50, 60, 70, 80, 100])
    age_rate = demo.groupby('AgeBin', observed=False)['SepsisLabel'].mean() * 100
    age_rate.plot(kind='bar', ax=axes[2], color='#f44336', edgecolor='black')
    axes[2].set_title('Sepsis Rate by Age Group', fontweight='bold')
    axes[2].set_ylabel('Sepsis Rate (%)')
    axes[2].set_xticklabels([str(x) for x in age_rate.index], rotation=45)

    plt.suptitle('Demographic Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demographics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: demographics.png")


def plot_correlation_heatmap(combined, output_dir):
    """Correlation heatmap of key clinical features."""
    key_feats = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN',
                 'Creatinine', 'Lactate', 'WBC', 'Platelets', 'Glucose', 'pH', 'SepsisLabel']
    sample = combined[key_feats].dropna()
    if len(sample) > 100000:
        sample = sample.sample(100000, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = sample.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap (Key Clinical Features)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")



if __name__ == "__main__":
    # Load data
    combined = load_all_patients(DATA_DIR)

    # Print summary
    print_dataset_summary(combined)

    # Save combined data
    combined.to_parquet(PARQUET_PATH, index=False)
    print(f"\nSaved combined data to: {PARQUET_PATH}")

    # Generate all visualizations
    print("\n--- Generating Visualizations ---")
    plot_missing_values_bar(combined, OUTPUT_DIR)
    plot_missing_values_heatmap(combined, OUTPUT_DIR)
    plot_class_distribution(combined, OUTPUT_DIR)
    plot_vital_signs_distributions(combined, OUTPUT_DIR)
    plot_icu_stay_distribution(combined, OUTPUT_DIR)
    plot_demographics(combined, OUTPUT_DIR)
    plot_correlation_heatmap(combined, OUTPUT_DIR)

    print("\nEDA complete! All figures saved to:", OUTPUT_DIR)
