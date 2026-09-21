"""
analyze_log.py
--------------
Generates visualization charts and evaluation statistics from outputs/detection_log.csv.
Produces plots for the college project report and presentation.

Usage:
    conda activate ipa
    python scripts/analyze_log.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze(csv_path="outputs/detection_log.csv", output_dir="outputs/visualizations"):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ Log file not found at '{csv_path}'. Run main.py or demo.py first!")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"📊 Visual Quality Inspection — Detection Log Analysis")
    print("=" * 60)

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} detection log entries from {csv_file}\n")

    if df.empty:
        print("⚠️ Log file is empty. No detections to analyze.")
        return

    # Print summary metrics
    print("--- Summary Metrics ---")
    print(f"Total Detections Logged: {len(df)}")
    print(f"Unique Frames with Defects: {df['frame'].nunique() if 'frame' in df.columns else 'N/A'}")
    if 'confidence' in df.columns:
        print(f"Average Confidence: {df['confidence'].mean():.3f} (Min: {df['confidence'].min():.3f}, Max: {df['confidence'].max():.3f})")
    
    if 'class' in df.columns:
        print("\nDetections by Class:")
        print(df['class'].value_counts().to_string())

    if 'severity' in df.columns:
        print("\nDetections by Severity:")
        print(df['severity'].value_counts().to_string())

    # Styling
    sns.set_theme(style="darkgrid")
    fig_size = (8, 5)

    # 1. Class Distribution Bar Chart
    if 'class' in df.columns:
        plt.figure(figsize=fig_size)
        ax = sns.countplot(data=df, x='class', palette='viridis', order=df['class'].value_counts().index)
        plt.title("Defect Class Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Defect / Object Class", fontsize=12)
        plt.ylabel("Detection Count", fontsize=12)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        chart1 = out_dir / "class_distribution.png"
        plt.savefig(chart1, dpi=300)
        plt.close()
        print(f"\n✅ Saved chart → {chart1}")

    # 2. Severity Distribution Pie Chart
    if 'severity' in df.columns:
        plt.figure(figsize=(6, 6))
        sev_counts = df['severity'].value_counts()
        colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c', 'critical': '#9b59b6'}
        pie_colors = [colors.get(s.lower(), '#3498db') for s in sev_counts.index]
        
        plt.pie(sev_counts, labels=sev_counts.index, autopct='%1.1f%%', startangle=140, colors=pie_colors,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title("Defect Severity Breakdown", fontsize=14, fontweight='bold')
        plt.tight_layout()
        chart2 = out_dir / "severity_distribution.png"
        plt.savefig(chart2, dpi=300)
        plt.close()
        print(f"✅ Saved chart → {chart2}")

    # 3. Confidence Distribution Histogram
    if 'confidence' in df.columns:
        plt.figure(figsize=fig_size)
        sns.histplot(df['confidence'], bins=15, kde=True, color='#3498db')
        plt.title("YOLO Detection Confidence Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Confidence Score (0 - 1.0)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        chart3 = out_dir / "confidence_distribution.png"
        plt.savefig(chart3, dpi=300)
        plt.close()
        print(f"✅ Saved chart → {chart3}")

    # 4. Defect Timeline Plot (Frame vs Confidence)
    if 'frame' in df.columns and 'confidence' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.scatterplot(data=df, x='frame', y='confidence', hue='severity' if 'severity' in df.columns else None,
                        style='class' if 'class' in df.columns else None, s=70)
        plt.title("Detection Timeline across Video Frames", fontsize=14, fontweight='bold')
        plt.xlabel("Frame Index", fontsize=12)
        plt.ylabel("Confidence Score", fontsize=12)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        chart4 = out_dir / "defect_timeline.png"
        plt.savefig(chart4, dpi=300)
        plt.close()
        print(f"✅ Saved chart → {chart4}")

    # 5. 2D FFT High-Frequency Energy Ratio by Class (Tier 1 Research)
    if 'high_freq_ratio' in df.columns and 'class' in df.columns:
        plt.figure(figsize=fig_size)
        sns.boxplot(data=df, x='class', y='high_freq_ratio', palette='mako')
        sns.stripplot(data=df, x='class', y='high_freq_ratio', color='orange', alpha=0.6, jitter=0.2)
        plt.title("2D FFT High-Frequency Ratio Across Defect Classes", fontsize=14, fontweight='bold')
        plt.xlabel("Defect Class", fontsize=12)
        plt.ylabel("High-Frequency Energy Ratio", fontsize=12)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        chart5 = out_dir / "fft_frequency_ratio_by_class.png"
        plt.savefig(chart5, dpi=300)
        plt.close()
        print(f"✅ Saved chart → {chart5}")

    # 6. VLM Uncertainty Distribution & Human Review Breakdown (Tier 1 Research)
    if 'uncertainty_score' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Subplot A: Histogram of Uncertainty Scores
        sns.histplot(df['uncertainty_score'], bins=12, kde=True, color='#e74c3c', ax=axes[0])
        axes[0].axvline(0.55, color='darkred', linestyle='--', label='Human Review Threshold (U >= 0.55)')
        axes[0].set_title("VLM Uncertainty Score Distribution", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Uncertainty Score (0 = Certain, 1 = Uncertain)", fontsize=11)
        axes[0].set_ylabel("Frequency", fontsize=11)
        axes[0].legend()

        # Subplot B: Confidence Labels & Review Breakdown
        if 'confidence_label' in df.columns:
            df['confidence_label'].value_counts().plot.pie(
                ax=axes[1],
                autopct='%1.1f%%',
                colors=['#2ecc71', '#f39c12', '#e74c3c'],
                startangle=140,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            axes[1].set_ylabel('')
            axes[1].set_title("Calibrated Confidence Breakdown", fontsize=13, fontweight='bold')

        plt.tight_layout()
        chart6 = out_dir / "vlm_uncertainty_analysis.png"
        plt.savefig(chart6, dpi=300)
        plt.close()
        print(f"✅ Saved chart → {chart6}")

    print("\n🎉 Analysis complete! All research charts generated for report.")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/detection_log.csv"
    analyze(csv_file)
