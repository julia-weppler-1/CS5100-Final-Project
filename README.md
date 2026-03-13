# CS5100-Final-Project
This project is about minimum sensor placement for reconstruction of Dissolved Oxygen observations in the Gulf of Mexico

## Instructions:
### Download data
Download the data as a CSV file from https://gisdata.gcoos.org/datasets/2e0f8cf7328c4f94a1377ed2b7469719_0/about and put it in a ```/data/``` folder in the root of this project. Rename it to ```DO.csv```

### Install requirements
This project relies on Python and uses DuckDB to handle the large dataset, as well as pandas and numpy, and scikit-learn for the linear regression baseline. These will need to be installed using requirements.txt.

### Run the code
You can run the files in the following order:
1. build_db.py
2. data_discovery.py
3. create_data_subset.py
4. baseline_linear_regression.py
These files will output data files (parquets) that subsequent files will rely on and will print artifacts to the terminal to better udnerstand the data.
