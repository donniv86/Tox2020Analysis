#!/usr/bin/env python3
"""
Streamlit Web Interface for Toxicity Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append('src')

from data_processing import Tox21DataLoader
from feature_engineering import MolecularFeatureGenerator
from models import ToxicityPredictor


def load_or_train_models():
    """Load pre-trained models or train new ones."""
    if os.path.exists('models/random_forest_models.pkl'):
        predictor = ToxicityPredictor('random_forest')
        predictor.load_models('models/random_forest_models.pkl')
        return predictor
    else:
        st.warning("No pre-trained models found. Training new models...")
        return train_models()


def train_models():
    """Train models on Tox21 data."""
    with st.spinner("Training models on Tox21 data..."):
        # Load Tox21 data
        loader = Tox21DataLoader("data/tox21_10k_data_all.sdf")
        data = loader.load_data()

        # Generate features
        feature_gen = MolecularFeatureGenerator()
        X = feature_gen.generate_features(
            data['compounds']['smiles'].tolist(),
            use_morgan=True,
            use_maccs=True,
            use_descriptors=True
        )

        # Split data
        splits = loader.split_data(test_size=0.2, random_state=42)

        X_train = feature_gen.generate_features(
            splits['train']['compounds']['smiles'].tolist(),
            use_morgan=True,
            use_maccs=True,
            use_descriptors=True
        )

        y_train = splits['train']['targets']

        # Train model
        predictor = ToxicityPredictor('random_forest')
        predictor.train(X_train, y_train)

        # Save models
        os.makedirs('models', exist_ok=True)
        predictor.save_models('models/random_forest_models.pkl')

        return predictor


def classify_toxicity(probabilities):
    """Classify toxicity based on probability thresholds."""
    classifications = []
    for prob in probabilities:
        if prob < 0.3:
            classifications.append('Non-toxic')
        elif prob < 0.7:
            classifications.append('Moderate')
        else:
            classifications.append('Toxic')
    return classifications


def create_toxicity_heatmap(results_df):
    """Create toxicity probability heatmap."""
    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    prob_columns = [f'{target}_probability' for target in target_columns
                   if f'{target}_probability' in results_df.columns]

    if prob_columns:
        prob_data = results_df[prob_columns].values
        compound_ids = results_df['compound_id'].tolist()
        target_names = [col.replace('_probability', '') for col in prob_columns]

        fig = px.imshow(
            prob_data,
            x=target_names,
            y=compound_ids,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            title='Toxicity Probability Heatmap'
        )

        fig.update_layout(
            xaxis_title='Toxicity Targets',
            yaxis_title='Compounds',
            height=400
        )

        return fig

    return None


def create_toxicity_summary_chart(results_df):
    """Create toxicity summary chart."""
    target_columns = [
        'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
        'SR-MMP', 'SR-p53'
    ]

    # Count toxicities per target
    target_summary = []
    for target in target_columns:
        tox_col = f'{target}_toxicity'
        if tox_col in results_df.columns:
            toxicity_counts = results_df[tox_col].value_counts()
            for toxicity, count in toxicity_counts.items():
                target_summary.append({
                    'Target': target,
                    'Toxicity': toxicity,
                    'Count': count
                })

    if target_summary:
        df_summary = pd.DataFrame(target_summary)

        fig = px.bar(
            df_summary,
            x='Target',
            y='Count',
            color='Toxicity',
            color_discrete_map={
                'Non-toxic': 'green',
                'Moderate': 'orange',
                'Toxic': 'red'
            },
            title='Toxicity Distribution by Target'
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )

        return fig

    return None


def main():
    st.set_page_config(
        page_title="Toxicity Prediction System",
        page_icon="🧪",
        layout="wide"
    )

    st.title("🧪 Toxicity Prediction System")
    st.markdown("Predict toxicity across 12 Tox21 endpoints using machine learning")

    # Sidebar
    st.sidebar.header("Input Options")

    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Single SMILES", "Upload CSV", "Example Compounds"]
    )

    # Load models
    predictor = load_or_train_models()

    if input_method == "Single SMILES":
        st.header("Single Compound Analysis")

        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        )

        if st.button("Predict Toxicity") and smiles_input:
            with st.spinner("Analyzing compound..."):
                # Create dataframe
                df = pd.DataFrame({
                    'compound_id': ['input_compound'],
                    'smiles': [smiles_input]
                })

                # Generate features
                feature_gen = MolecularFeatureGenerator()
                X = feature_gen.generate_features(
                    df['smiles'].tolist(),
                    use_morgan=True,
                    use_maccs=True,
                    use_descriptors=True
                )

                # Make predictions
                predictions = predictor.predict(X)

                # Process results
                results = df.copy()
                target_columns = [
                    'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
                    'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                    'SR-MMP', 'SR-p53'
                ]

                for target in target_columns:
                    if target in predictions.columns:
                        results[f'{target}_probability'] = predictions[target]
                        results[f'{target}_prediction'] = (predictions[target] > 0.5).astype(int)
                        results[f'{target}_toxicity'] = classify_toxicity(predictions[target])

                # Display results
                st.subheader("Toxicity Predictions")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Compound Details:**")
                    st.write(f"SMILES: {smiles_input}")

                    # Overall summary
                    toxic_count = 0
                    moderate_count = 0
                    non_toxic_count = 0

                    for target in target_columns:
                        tox_col = f'{target}_toxicity'
                        if tox_col in results.columns:
                            toxicity = results.iloc[0][tox_col]
                            if toxicity == 'Toxic':
                                toxic_count += 1
                            elif toxicity == 'Moderate':
                                moderate_count += 1
                            else:
                                non_toxic_count += 1

                    st.write(f"**Overall Summary:**")
                    st.write(f"🟢 Non-toxic: {non_toxic_count} targets")
                    st.write(f"🟡 Moderate: {moderate_count} targets")
                    st.write(f"🔴 Toxic: {toxic_count} targets")

                    if toxic_count > 0:
                        st.error("⚠️ WARNING: This compound shows toxic properties!")
                    elif moderate_count > 0:
                        st.warning("⚠️ CAUTION: This compound shows moderate toxicity.")
                    else:
                        st.success("✅ SAFE: This compound appears non-toxic.")

                with col2:
                    # Target-wise results
                    st.write("**Target-wise Results:**")

                    for target in target_columns:
                        tox_col = f'{target}_toxicity'
                        prob_col = f'{target}_probability'

                        if tox_col in results.columns and prob_col in results.columns:
                            toxicity = results.iloc[0][tox_col]
                            probability = results.iloc[0][prob_col]

                            if toxicity == 'Toxic':
                                st.write(f"🔴 {target}: {toxicity} ({probability:.3f})")
                            elif toxicity == 'Moderate':
                                st.write(f"🟡 {target}: {toxicity} ({probability:.3f})")
                            else:
                                st.write(f"🟢 {target}: {toxicity} ({probability:.3f})")

                # Create visualizations
                st.subheader("Visualizations")

                # Probability heatmap
                heatmap_fig = create_toxicity_heatmap(results)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

    elif input_method == "Upload CSV":
        st.header("Batch Analysis")

        uploaded_file = st.file_uploader(
            "Upload CSV file with SMILES:",
            type=['csv'],
            help="CSV should have 'compound_id' and 'smiles' columns"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(df.head())

                if st.button("Analyze Compounds"):
                    with st.spinner("Analyzing compounds..."):
                        # Validate data
                        if 'smiles' not in df.columns:
                            st.error("CSV must contain a 'smiles' column!")
                            return

                        if 'compound_id' not in df.columns:
                            df['compound_id'] = [f'compound_{i+1}' for i in range(len(df))]

                        # Generate features
                        feature_gen = MolecularFeatureGenerator()
                        X = feature_gen.generate_features(
                            df['smiles'].tolist(),
                            use_morgan=True,
                            use_maccs=True,
                            use_descriptors=True
                        )

                        # Make predictions
                        predictions = predictor.predict(X)

                        # Process results
                        results = df.copy()
                        target_columns = [
                            'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
                            'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                            'SR-MMP', 'SR-p53'
                        ]

                        for target in target_columns:
                            if target in predictions.columns:
                                results[f'{target}_probability'] = predictions[target]
                                results[f'{target}_prediction'] = (predictions[target] > 0.5).astype(int)
                                results[f'{target}_toxicity'] = classify_toxicity(predictions[target])

                        # Display results
                        st.subheader("Analysis Results")

                        # Summary statistics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Compounds", len(results))

                        with col2:
                            toxic_compounds = 0
                            for idx, row in results.iterrows():
                                toxic_count = sum(1 for target in target_columns
                                                if f'{target}_toxicity' in row and row[f'{target}_toxicity'] == 'Toxic')
                                if toxic_count > 0:
                                    toxic_compounds += 1
                            st.metric("Toxic Compounds", toxic_compounds)

                        with col3:
                            safe_compounds = len(results) - toxic_compounds
                            st.metric("Safe Compounds", safe_compounds)

                        # Results table
                        st.write("**Detailed Results:**")
                        display_cols = ['compound_id', 'smiles']
                        for target in target_columns:
                            display_cols.extend([f'{target}_toxicity', f'{target}_probability'])

                        available_cols = [col for col in display_cols if col in results.columns]
                        st.dataframe(results[available_cols])

                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="toxicity_predictions.csv",
                            mime="text/csv"
                        )

                        # Visualizations
                        st.subheader("Visualizations")

                        col1, col2 = st.columns(2)

                        with col1:
                            heatmap_fig = create_toxicity_heatmap(results)
                            if heatmap_fig:
                                st.plotly_chart(heatmap_fig, use_container_width=True)

                        with col2:
                            summary_fig = create_toxicity_summary_chart(results)
                            if summary_fig:
                                st.plotly_chart(summary_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif input_method == "Example Compounds":
        st.header("Example Analysis")

        example_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Gefitinib
            "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        ]

        example_names = ["Ibuprofen", "Gefitinib", "Celecoxib"]

        df = pd.DataFrame({
            'compound_id': example_names,
            'smiles': example_smiles
        })

        st.write("**Example Compounds:**")
        st.dataframe(df)

        if st.button("Analyze Example Compounds"):
            with st.spinner("Analyzing example compounds..."):
                # Generate features
                feature_gen = MolecularFeatureGenerator()
                X = feature_gen.generate_features(
                    df['smiles'].tolist(),
                    use_morgan=True,
                    use_maccs=True,
                    use_descriptors=True
                )

                # Make predictions
                predictions = predictor.predict(X)

                # Process results
                results = df.copy()
                target_columns = [
                    'NR-Aromatase', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD',
                    'NR-PPAR-gamma', 'NR-AhR', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                    'SR-MMP', 'SR-p53'
                ]

                for target in target_columns:
                    if target in predictions.columns:
                        results[f'{target}_probability'] = predictions[target]
                        results[f'{target}_prediction'] = (predictions[target] > 0.5).astype(int)
                        results[f'{target}_toxicity'] = classify_toxicity(predictions[target])

                # Display results
                st.subheader("Example Results")

                for idx, row in results.iterrows():
                    st.write(f"**{row['compound_id']}**")

                    toxic_count = 0
                    moderate_count = 0
                    non_toxic_count = 0

                    for target in target_columns:
                        tox_col = f'{target}_toxicity'
                        prob_col = f'{target}_probability'

                        if tox_col in row and prob_col in row:
                            toxicity = row[tox_col]
                            probability = row[prob_col]

                            if toxicity == 'Toxic':
                                toxic_count += 1
                                st.write(f"🔴 {target}: {toxicity} ({probability:.3f})")
                            elif toxicity == 'Moderate':
                                moderate_count += 1
                                st.write(f"🟡 {target}: {toxicity} ({probability:.3f})")
                            else:
                                non_toxic_count += 1
                                st.write(f"🟢 {target}: {toxicity} ({probability:.3f})")

                    st.write(f"**Summary:** {toxic_count} Toxic, {moderate_count} Moderate, {non_toxic_count} Non-toxic")
                    st.write("---")

                # Visualizations
                st.subheader("Example Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    heatmap_fig = create_toxicity_heatmap(results)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                with col2:
                    summary_fig = create_toxicity_summary_chart(results)
                    if summary_fig:
                        st.plotly_chart(summary_fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About:** This system predicts toxicity across 12 Tox21 endpoints using machine learning models trained on the Tox21 dataset.

        **Toxicity Classification:**
        - 🟢 Non-toxic: Probability < 0.3
        - 🟡 Moderate: Probability 0.3-0.7
        - 🔴 Toxic: Probability > 0.7
        """
    )


if __name__ == "__main__":
    main()