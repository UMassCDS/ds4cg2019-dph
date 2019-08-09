# ds4cg2019-dph

### Setup and Running Code
#### Install Requirements
1. Run ```pip freeze > requirements.txt```
2. Run ```pip install -r requirements.txt```

#### Run Code
1. After setting file variables in ```run.py```, run the code using ```python run.py```
2. For visualizations of correlation matrices, bar charts for health scores for Top 3 and Bottom 3 communities, ```python visualize.py```
3. For visualizations of specific communities, ```python visualize_communities.py```

### data 
Folder that contains all data that is used as the input for this process.

### output 
Folder that contains files that reflect the different outputs for this process.
This files only contain numbers and values, they do not convey semantics.

### visualizations
Folder that contains visualizations of the different files within the output folder.

##result_projections
Folder that contains D3 code to visualize the results of the composite Health Score, and project them onto a map of the state of Massachusetts.

###build_fulldataset.py
Combines determinant data and outcome data into one data set that contains all the data.

###clean.py
Cleans the given dataset by converting totals to rates/percentages, removing sparse columns, normalizing percentages, and coerce strings with special characters to numbers. (i.e. "1,000,000+" => 1,000,000)

###impute.py
Imputes the missing values within a given dataset.

###pca.py
Establishes the HealthScores class.
This class is responsible for all the functions related to producing composite health scores based off a PCA process.
Factor and Correlation Analysis are also present within the class.

###run.py
Produces the results seen in the output folder using files from the data folder as input.

###visualize_communities.py
Produces two histograms (one for subsets, one for domains) displaying results across all different data types of data for a specific given community.

###visualize.py
Produces the rest of the visualizations within the visualizations folder except for the ones made within results_projections.
Correlation matricies, plotting comparisons, and top 3 vs bottom 3 comparisons.

