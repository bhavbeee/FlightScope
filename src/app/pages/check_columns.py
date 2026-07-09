# import pandas as pd

# # Change this to your flight CSV file
# file = r"C:\Users\jiyaa\OneDrive\Desktop\CS661_data\data\processed\airports.csv"

# df = pd.read_csv(file)

# print(df.columns.tolist())

# print("\nFirst 5 rows:")
# print(df.head())

import os
import pandas as pd


folder = r"C:\Users\jiyaa\OneDrive\Desktop\CS661_data\data\processed"


for file in os.listdir(folder):

    if file.endswith(".csv"):

        path = os.path.join(folder, file)

        print("\n==============================")
        print("FILE:", file)

        try:
            df = pd.read_csv(path)

            print("Columns:")
            print(df.columns.tolist())

            print("Rows:", len(df))

        except Exception as e:
            print("Error:", e)