# CS5100-Final-Project
This project is about minimum sensor placement for reconstruction of Dissolved Oxygen observations in the Gulf of Mexico

## Instructions:
### Download data
Download the data as a CSV file from https://gisdata.gcoos.org/datasets/2e0f8cf7328c4f94a1377ed2b7469719_0/about and put it in a ```/data/``` folder in the root of this project. Rename it to ```DO.csv```

### Install requirements
This project relies on Python and uses DuckDB to handle the large dataset, as well as pandas, numpy, scikit-learn for machine learning, pyarrow for reading and writing parquet files, and matplotlib for visualizations. These will need to be installed using pip install numpy pandas scikit-learn pyarrow duckdb matplotlib.

### Run the code
You can run the files in the following order:
1. python build_db.py
2. python create_data_subset.py
3. python stage1_build_matrix.py
4. python stage2_reconstruction.py
5. python stage3_genetic_algorithm.py
6. python stage4_visualize.py
These files will output data files (parquets) that subsequent files will rely on and will print artifacts to the terminal to better udnerstand the data.
