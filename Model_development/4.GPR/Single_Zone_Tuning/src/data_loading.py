import pandas as pd
from constants import DATA_PATH, DATE_COL, ZONE_ID_COL

def load_data(zone_id, data_path=DATA_PATH):
    """
    Load and filter wildfire dataset for a specific zone.
    """
    df = pd.read_csv(data_path, parse_dates=[DATE_COL])
    df = df[df[ZONE_ID_COL] == zone_id].sort_values(by=DATE_COL).reset_index(drop=True)
    return df


