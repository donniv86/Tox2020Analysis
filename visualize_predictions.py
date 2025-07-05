#!/usr/bin/env python3
"""
Visualization script for Tox21 toxicity predictions.
Generates comprehensive plots showing which molecules passed or failed each toxicity target.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ToxicityVisualizer:
    """Visualize toxicity predictions for molecules."""

    def __init__(self, results_file='test_folder/toxicity_predictions.csv'):
        self.results_file = results_file
        self.df = None
        self.targets = []
        self.load_results()

    def load_results(self):
        """Load prediction results from CSV."""
        if not os.path.exists(self.results_file):
            print(f"Results file {self.results_file} not found!")
            return

        self.df = pd.read_csv(self.results_file)
        print(f"Loaded {len(self.df)} molecules with predictions")

        # Extract target names from column names
        self.targets = []
        for col in self.df.columns:
            if col.endswith('_prediction'):
                target = col.replace('_prediction', '')
                self.targets.append(target)

        print(f"Found {len(self.targets)} targets: {self.targets}")

    def create_summary_plot(self, pdf_pages):
        """Create overall summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tox21 Toxicity Prediction Summary', fontsize=16, fontweight='bold')

        # 1. Pass/Fail counts for each target
        ax1 = axes[0, 0]
        pass_fail_counts = []
        target_names = []

        for target in self.targets:
            pred_col = f'{target}_prediction'
            if pred_col in self.df.columns:
                passed = (self.df[pred_col] == 0).sum()
                failed = (self.df[pred_col] == 1).sum()
                pass_fail_counts.append([passed, failed])
                target_names.append(target)

        if pass_fail_counts:
            pass_fail_df = pd.DataFrame(pass_fail_counts,
                                      columns=['Passed', 'Failed'],
                                      index=target_names)
            pass_fail_df.plot(kind='bar', ax=ax1, color=['green', 'red'])
            ax1.set_title('Molecules Passed vs Failed by Target')
            ax1.set_ylabel('Number of Molecules')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()

        # 2. Average probability by target
        ax2 = axes[0, 1]
        avg_probs = []
        for target in self.targets:
            prob_col = f'{target}_probability'
            if prob_col in self.df.columns:
                avg_prob = self.df[prob_col].mean()
                avg_probs.append(avg_prob)
            else:
                avg_probs.append(0)

        bars = ax2.bar(target_names, avg_probs, color='skyblue', alpha=0.7)
        ax2.set_title('Average Toxicity Probability by Target')
        ax2.set_ylabel('Average Probability')
        ax2.tick_params(axis='x', rotation=45)

        # Add probability values on bars
        for bar, prob in zip(bars, avg_probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. Distribution of probabilities
        ax3 = axes[1, 0]
        all_probs = []
        for target in self.targets:
            prob_col = f'{target}_probability'
            if prob_col in self.df.columns:
                all_probs.extend(self.df[prob_col].values)

        if all_probs:
            ax3.hist(all_probs, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax3.set_title('Distribution of All Toxicity Probabilities')
            ax3.set_xlabel('Probability')
            ax3.set_ylabel('Frequency')
            ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
            ax3.legend()

        # 4. Heatmap of probabilities
        ax4 = axes[1, 1]
        prob_matrix = []
        for target in self.targets:
            prob_col = f'{target}_probability'
            if prob_col in self.df.columns:
                prob_matrix.append(self.df[prob_col].values)

        if prob_matrix:
            prob_df = pd.DataFrame(prob_matrix, index=target_names).T
            sns.heatmap(prob_df, ax=ax4, cmap='RdYlGn_r', center=0.5,
                       cbar_kws={'label': 'Probability'})
            ax4.set_title('Toxicity Probability Heatmap')
            ax4.set_xlabel('Targets')
            ax4.set_ylabel('Molecules')

        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close()

    def create_target_detail_plots(self, pdf_pages):
        """Create detailed plots for each target."""
        for target in self.targets:
            pred_col = f'{target}_prediction'
            prob_col = f'{target}_probability'

            if pred_col not in self.df.columns or prob_col not in self.df.columns:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Toxicity Analysis: {target}', fontsize=16, fontweight='bold')

            # 1. Probability distribution
            ax1 = axes[0, 0]
            probabilities = self.df[prob_col]
            passed_probs = probabilities[self.df[pred_col] == 0]
            failed_probs = probabilities[self.df[pred_col] == 1]

            ax1.hist(passed_probs, bins=20, alpha=0.7, label='Passed', color='green')
            ax1.hist(failed_probs, bins=20, alpha=0.7, label='Failed', color='red')
            ax1.set_title(f'Probability Distribution for {target}')
            ax1.set_xlabel('Probability')
            ax1.set_ylabel('Frequency')
            ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
            ax1.legend()

            # 2. Pass/Fail pie chart
            ax2 = axes[0, 1]
            passed_count = (self.df[pred_col] == 0).sum()
            failed_count = (self.df[pred_col] == 1).sum()

            if passed_count + failed_count > 0:
                sizes = [passed_count, failed_count]
                labels = ['Passed', 'Failed']
                colors = ['green', 'red']
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'Pass/Fail Distribution for {target}')

            # 3. Top 10 most toxic molecules
            ax3 = axes[1, 0]
            top_toxic = self.df.nlargest(10, prob_col)[['Smiles', prob_col]]
            if len(top_toxic) > 0:
                # Truncate SMILES for display
                top_toxic['Smiles_short'] = top_toxic['Smiles'].str[:30] + '...'
                bars = ax3.barh(range(len(top_toxic)), top_toxic[prob_col], color='red', alpha=0.7)
                ax3.set_yticks(range(len(top_toxic)))
                ax3.set_yticklabels(top_toxic['Smiles_short'])
                ax3.set_title(f'Top 10 Most Toxic Molecules for {target}')
                ax3.set_xlabel('Probability')

                # Add probability values
                for i, (bar, prob) in enumerate(zip(bars, top_toxic[prob_col])):
                    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{prob:.3f}', ha='left', va='center', fontsize=8)

            # 4. Statistics summary
            ax4 = axes[1, 1]
            ax4.axis('off')

            stats_text = f"""
            Target: {target}

            Total Molecules: {len(self.df)}
            Passed: {passed_count} ({passed_count/len(self.df)*100:.1f}%)
            Failed: {failed_count} ({failed_count/len(self.df)*100:.1f}%)

            Probability Statistics:
            Mean: {probabilities.mean():.3f}
            Median: {probabilities.median():.3f}
            Std: {probabilities.std():.3f}
            Min: {probabilities.min():.3f}
            Max: {probabilities.max():.3f}

            Threshold: 0.5
            """

            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close()

    def create_molecule_detail_plots(self, pdf_pages):
        """Create detailed plots for individual molecules."""
        # Select top 20 molecules with highest average toxicity
        avg_probs = []
        for i, row in self.df.iterrows():
            probs = []
            for target in self.targets:
                prob_col = f'{target}_probability'
                if prob_col in self.df.columns:
                    probs.append(row[prob_col])
            avg_probs.append(np.mean(probs))

        self.df['avg_toxicity'] = avg_probs
        top_molecules = self.df.nlargest(20, 'avg_toxicity')

        for idx, (_, molecule) in enumerate(top_molecules.iterrows()):
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Molecule {idx+1}: {molecule["Smiles"][:50]}...',
                        fontsize=14, fontweight='bold')

            # 1. Toxicity profile across all targets
            ax1 = axes[0]
            target_probs = []
            target_names = []

            for target in self.targets:
                prob_col = f'{target}_probability'
                if prob_col in self.df.columns:
                    target_probs.append(molecule[prob_col])
                    target_names.append(target)

            bars = ax1.bar(target_names, target_probs,
                          color=['red' if p > 0.5 else 'green' for p in target_probs],
                          alpha=0.7)
            ax1.set_title('Toxicity Profile Across All Targets')
            ax1.set_ylabel('Probability')
            ax1.tick_params(axis='x', rotation=45)
            ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold')
            ax1.legend()

            # Add probability values on bars
            for bar, prob in zip(bars, target_probs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

            # 2. Summary statistics
            ax2 = axes[1]
            ax2.axis('off')

            # Count passed/failed targets
            passed_targets = sum(1 for p in target_probs if p <= 0.5)
            failed_targets = sum(1 for p in target_probs if p > 0.5)

            summary_text = f"""
            Molecule Summary:

            SMILES: {molecule['Smiles']}
            Average Toxicity: {molecule['avg_toxicity']:.3f}

            Targets Passed: {passed_targets}/{len(target_names)}
            Targets Failed: {failed_targets}/{len(target_names)}

            Highest Risk Targets:
            """

            # Find top 3 most toxic targets
            target_prob_pairs = list(zip(target_names, target_probs))
            target_prob_pairs.sort(key=lambda x: x[1], reverse=True)

            for i, (target, prob) in enumerate(target_prob_pairs[:3]):
                summary_text += f"\n{i+1}. {target}: {prob:.3f}"

            ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close()

    def generate_all_plots(self, output_pdf='test_folder/toxicity_analysis.pdf'):
        """Generate all visualization plots and save to PDF."""
        if self.df is None:
            print("No data loaded!")
            return

        print(f"Generating comprehensive toxicity analysis plots...")
        print(f"Output will be saved to: {output_pdf}")

        with PdfPages(output_pdf) as pdf_pages:
            # 1. Summary plots
            print("Creating summary plots...")
            self.create_summary_plot(pdf_pages)

            # 2. Target detail plots
            print("Creating target detail plots...")
            self.create_target_detail_plots(pdf_pages)

            # 3. Molecule detail plots
            print("Creating molecule detail plots...")
            self.create_molecule_detail_plots(pdf_pages)

        print(f"✅ All plots generated and saved to: {output_pdf}")
        print(f"📊 Total pages: {len(self.targets) + 3}")  # Summary + targets + molecules

def main():
    """Main function to generate toxicity visualization plots."""
    print("=" * 60)
    print("TOX21 TOXICITY PREDICTION VISUALIZATION")
    print("=" * 60)

    # Initialize visualizer
    visualizer = ToxicityVisualizer()

    if visualizer.df is None:
        print("❌ Failed to load prediction results!")
        return

    # Generate all plots
    output_file = 'test_folder/toxicity_analysis.pdf'
    visualizer.generate_all_plots(output_file)

    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETED!")
    print("=" * 60)
    print(f"📄 PDF Report: {output_file}")
    print(f"📊 Molecules Analyzed: {len(visualizer.df)}")
    print(f"🎯 Targets Analyzed: {len(visualizer.targets)}")
    print("\nThe PDF contains:")
    print("  • Overall summary plots")
    print("  • Detailed analysis for each target")
    print("  • Individual molecule profiles")
    print("  • Pass/Fail statistics")

if __name__ == "__main__":
    main()