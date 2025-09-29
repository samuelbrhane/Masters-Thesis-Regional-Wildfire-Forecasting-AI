import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Cleaned_Data", "zone_sequence_merged.csv")

FEATURE_COLS = ['Temperature', 'Precipitation', 'Humidity', 'Wind', 'Prev_Num_Fires_Result']
TARGET_COL = 'Num_Fires'
DATE_COL = 'Date'
ZONE_ID_COL = 'Zone_ID'

SEED = 42 
EARLY_STOPPING_PATIENCE = 10  
MAX_EPOCHS = 50
LOSS_FUNCTION = 'mse'
  
