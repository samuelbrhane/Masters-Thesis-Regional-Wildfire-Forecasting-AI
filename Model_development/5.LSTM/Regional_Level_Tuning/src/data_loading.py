import pandas as pd
from constants import DATA_PATH, DATE_COL, ZONE_ID_COL

def load_data(zone_ids=None, data_path=DATA_PATH):
    """
    Load wildfire dataset, optionally filtering by one or more zone IDs.
    """
    df = pd.read_csv(data_path, parse_dates=[DATE_COL])

    if zone_ids is not None:
        if isinstance(zone_ids, int):
            zone_ids = [zone_ids]
        df = df[df[ZONE_ID_COL].isin(zone_ids)]

    df = df.drop(columns=[ZONE_ID_COL])

    aggregation = {
        'Precipitation': 'mean',
        'Humidity': 'mean',
        'Temperature': 'mean',
        'Wind': 'mean',
        'Num_Fires': 'sum'
    }

    df = df.groupby(DATE_COL).agg(aggregation).reset_index()
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)

    return df
