import pandas as pd

def read_process_csv(filename: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index("Timestamp")
    # Drop the $ from the monthly gold price, we know it is dollars per ounce
    df[colname] = df[colname].str.replace('$', '')
    df[colname] = df[colname].str.replace(',', '')
    df[colname] = df[colname].astype("float64")
    df = df.sort_index()
    return df