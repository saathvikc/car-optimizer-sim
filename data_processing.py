import pandas as pd

def load_csv_features(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Experiment", "Std Cd"], errors='ignore')
    X = df.drop(columns=["Average Cd"])
    y = df["Average Cd"]
    return X, y
