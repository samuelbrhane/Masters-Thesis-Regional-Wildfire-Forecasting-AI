# Data Preprocessing and Analysis

This folder contains all scripts used to prepare, process, and analyze the wildfire and climate datasets before modeling. It covers spatial mapping, climate variable extraction, fire occurrence aggregation, and exploratory data analysis.

## Contents

- **Warning Area Analysis (`1.WarningAreas_Analysis.ipynb`)**  
  - Loads *WarningAreas.shp*  
  - Maps grid cells to custom warning zones  
  - Saves zone mappings (`cell_zones.parquet`, shapefiles)  
  - Produces zone visualization plots  

- **Climate Analysis (`2.Climate_Analysis.ipynb`)**  
  - Extracts daily climate variables (precipitation, humidity, temperature, wind) from HDF5 files  
  - Maps climate data to zones  
  - Creates spatial distribution plots  

- **Target Analysis (`3.Target_Analysis.ipynb`)**  
  - Loads wildfire target grids (fire presence/absence)  
  - Maps fire cells to zones  
  - Generates wildfire occurrence plots per zone  

- **Single-Day Aggregation (`4.Single_Day_Aggregate.ipynb`)**  
  - Aggregates fire clusters and climate data at zone level for a single day  
  - Combines fire counts with average climate conditions  

- **Yearly Aggregation (`5.Yearly_Aggregate_Pipeline_Climate_Fire.ipynb`)**  
  - Automates daily processing across all files in a year  
  - Outputs per-zone daily CSVs with fire counts and climate averages  
  - Parallelized for efficiency  

- **Exploratory Data Analysis (`6.Data_Analysis_Climate_Fire.ipynb`)**  
  - Merges all yearly data into a single dataset (`zone_sequence_merged.csv`)  
  - Performs zone-level and region-level exploratory analysis  
  - Generates time series, distributions, seasonal trends, and correlation plots  
  - Stores figures inside `EDA_plots/`

---


