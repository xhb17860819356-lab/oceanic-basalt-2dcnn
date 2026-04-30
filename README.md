# Oceanic Basalt 2D-CNN

This repository contains the source code used for the manuscript:

**Petrogenetic Discrimination of Oceanic Basalts based on Convolutional Neural Network**

## Overview

This code implements a deep-learning workflow for the petrogenetic discrimination of oceanic basalts, including mid-ocean ridge basalts (MORB), ocean-island basalts (OIB), and island-arc basalts (IAB).

The workflow includes:

- Geochemical data preprocessing
- Construction of a 3×6 two-dimensional geochemical feature matrix
- Training of single 2D-CNN models
- Heterogeneous ensemble CNN optimization
- Random forest baseline comparison
- Model evaluation using accuracy, confusion matrices, precision, recall, and F1-score
- t-SNE visualization of CNN-extracted features
- SHAP-based interpretability analysis

## Data

The geochemical data used in this study were compiled from the public GEOROC and PetDB databases.

The processed dataset is not included in this repository due to data redistribution considerations.

To run the code, users should prepare the processed Excel file and place it in the following path:

`data/combined_for_DL_13000.xlsx`

The input file should contain the following feature columns:

`SiO2_wt`, `TiO2_wt`, `Al2O3_wt`, `FeO_wt`, `MgO_wt`, `CaO_wt`, `Na2O_wt`, `K2O_wt`, `P2O5_wt`, `Nb_ppm`, `Zr_ppm`, `Y_ppm`, `Th_ppm`, `Yb_ppm`, `Nb_Yb`, `Th_Yb`, `log_Nb_Yb`, `log_Th_Yb`

The class label column should be named:

`label`

The class labels should include:

`IAB`, `MORB`, `OIB`

The processed data supporting the findings of this study are available from the corresponding author upon reasonable request.

## Requirements

The code was developed in Python and requires the following packages:

- numpy
- pandas
- openpyxl
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- shap

Install the required Python packages using:

`pip install -r requirements.txt`

## Usage

Run the main script:

`python basalt_2dcnn.py`

The script performs data preprocessing, feature matrix construction, model training, model evaluation, visualization, and SHAP-based interpretability analysis.

## Outputs

The script generates the following results and figures:

- Pearson correlation heatmap
- 3×6 geochemical feature matrix layout
- Random forest feature importance ranking
- CNN training curve
- t-SNE visualization of CNN-extracted features
- Confusion matrices and normalized confusion matrices
- Model accuracy comparison
- Class-wise precision, recall, and F1-score comparison
- Global SHAP summary plot
- Class-wise SHAP feature-importance plots

Figures are saved in PDF, EPS, and high-resolution PNG formats.

## Code Availability

This repository provides the source code for data preprocessing, two-dimensional geochemical feature encoding, 2D-CNN model training, heterogeneous ensemble learning, model evaluation, t-SNE visualization, and SHAP-based interpretability analysis.

## Notes

The processed dataset is not distributed with this repository. Users should prepare the required geochemical dataset according to the column format described above before running the code.