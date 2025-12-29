# data-science-visualization
Daily work and hands-on practice in Data Science and Visualization using Pandas, EDA, Machine Learning, Matplotlib and Seaborn.
Experiment 1 – Categorical Data Analysis using Pandas
Objective

To create and analyze categorical and ordered categorical data using Pandas and perform frequency and descriptive analysis.

Libraries Used
import pandas as pd
import numpy as np

Description

This experiment demonstrates how Pandas handles categorical variables, including ordered categories, and how to generate statistical summaries for datasets containing both categorical and numerical attributes.

Key Syntax Explanation

pd.Categorical()
Converts data into categorical format with an optional order.

value_counts()
Computes frequency of each category.

sort_index()
Sorts frequency output by category order.

describe(include='all')
Generates descriptive statistics for all columns.

dtypes
Displays data types of DataFrame columns.

info()
Provides dataset structure and memory usage.

Result

Successfully analyzed categorical variables using Pandas.

Experiment 2 – GroupBy Operations in Pandas
Objective

To perform aggregation, filtering, and transformation operations using Pandas GroupBy functionality.

Libraries Used
import pandas as pd
import numpy as np

Description

This experiment focuses on grouping data by categorical variables and applying statistical operations on each group.

Key Syntax Explanation

groupby()
Splits data into groups based on a key column.

aggregate()
Applies one or more aggregation functions to grouped data.

Column-wise aggregation
Allows different aggregation functions for different columns.

filter()
Filters entire groups based on a condition.

transform()
Performs group-wise transformation while preserving original data shape.

Result

Gained experience in advanced group-wise data analysis.

Experiment 3 – Cross Tabulation and Correlation Analysis
Objective

To analyze relationships between categorical and numerical variables using cross-tabulation and correlation techniques.

Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Description

This experiment uses the Toyota dataset to study relationships among variables through frequency tables, normalization, and correlation visualization.

Key Syntax Explanation

read_csv()
Loads dataset and handles missing values.

pd.crosstab()
Creates frequency tables between categorical variables.

normalize=True
Converts frequencies to probabilities.

margins=True
Adds row and column totals.

select_dtypes()
Selects numerical columns only.

corr()
Computes correlation matrix.

Result

Identified relationships between variables using cross-tabulation and correlation analysis.

Experiment 4 – Exploratory Data Analysis (EDA)
Objective

To visually explore datasets and identify distributions, trends, and outliers.

Libraries Used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Description

This experiment applies visual techniques to understand data behavior and detect anomalies.

Key Syntax Explanation

Histogram
Visualizes data distribution.

Heatmap
Displays correlation intensity.

Pairplot
Shows pairwise relationships.

Box plot
Detects outliers and spread.

Result

Improved understanding of data patterns through visualization.

Experiment 5 – Model Building for Data Analytics (Iris Dataset)
Objective

To build and evaluate a classification model using Logistic Regression.

Libraries Used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

Description

This experiment involves preprocessing, visualization, model training, and evaluation using the Iris dataset.

Key Syntax Explanation

LabelEncoder
Converts categorical labels into numerical form.

train_test_split()
Splits dataset into training and testing sets.

LogisticRegression()
Builds a classification model.

fit() and predict()
Trains the model and generates predictions.

Evaluation metrics
Accuracy, Precision, Recall, and F1 Score measure model performance.

Result

Achieved high classification accuracy using Logistic Regression.
