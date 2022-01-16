# UCI-Heart-dataset-Analysis
Performed Data cleaning and EDA analysis for the UCI Heart dataset, looking at various methods

## Abstract

The task is to perform data cleaning on the given heart.csv dataset.The dataset is procured from the [UCI Heart disease dataset](https://www.kaggle.com/ronitf/heart-disease-uci). The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 1. In the process following aspects of the data are checked - 

- Identifying Outliers and removing if any
- Imputing missing values ie., look at multiple methods for imputing categorical, numeric, and time-series data
- Adjusts for bias in the dataset
- Compare and contrast metrics to observe what changed from Original to changed dataset
- Visualize the metrics to observe what changed from Original to changed dataset


In order to perform the task, first the EDA (using seaborn/matplotlib) was done to get a sense of the data. Later on various techniques of data cleaning are applied to the dataset and/or suggested for any other type of dataset as well(say dataset had time-series).


```
Acknowledgements

Creators:

    Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Donor:
David W. Aha (aha '@' ics.uci.edu) (714) 856-8779 
```
