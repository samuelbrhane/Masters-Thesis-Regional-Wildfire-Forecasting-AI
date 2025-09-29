DATA_PATH = "../Cleaned_Data/zone_sequence_merged.csv"  
RESULTS_DIR = "results/"
MODELS_DIR = "models/"

FEATURE_COLS = ['Temperature', 'Precipitation', 'Humidity', 'Wind', 'Prev_Num_Fires_Result']
TARGET_COL = 'Num_Fires'
DATE_COL = 'Date'
ZONE_ID_COL = 'Zone_ID'

SEED = 42 
EARLY_STOPPING_PATIENCE = 10  
MAX_EPOCHS = 50
LOSS_FUNCTION = 'mse'
  
