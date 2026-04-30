# Data

The processed geochemical dataset used in this study is not included in this repository.

The data were compiled from the public GEOROC and PetDB databases. Due to data redistribution considerations, the processed dataset is available from the corresponding author upon reasonable request.

To run the code, users should prepare an Excel file named:

`combined_for_DL_13000.xlsx`

and place it in this folder.

The input file should contain the following feature columns:

`SiO2_wt`, `TiO2_wt`, `Al2O3_wt`, `FeO_wt`, `MgO_wt`, `CaO_wt`, `Na2O_wt`, `K2O_wt`, `P2O5_wt`, `Nb_ppm`, `Zr_ppm`, `Y_ppm`, `Th_ppm`, `Yb_ppm`, `Nb_Yb`, `Th_Yb`, `log_Nb_Yb`, `log_Th_Yb`

The class label column should be named:

`label`

The class labels should include:

`IAB`, `MORB`, `OIB`
