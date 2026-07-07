import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_feature_engineering():
    input_path = "Flights_2022_sampled_1.8M.parquet"
    output_parquet = "data/processed_flights.parquet"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file {input_path} not found in current directory.")
        
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 1. Clean column names (strip trailing spaces)
    df.columns = df.columns.str.strip()
    print("Columns sanitized.")
    
    # 2. Time and Date Features
    print("Engineering time-based features...")
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([4, 5, 6]).astype(int)
    df['Month'] = df['FlightDate'].dt.month
    df['DayOfMonth'] = df['FlightDate'].dt.day
    
    # Season: 1=Winter (Dec, Jan, Feb), 2=Spring (Mar, Apr, May), 3=Summer (Jun, Jul, Aug), 4=Fall (Sep, Oct, Nov)
    df['Season'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))
    
    # Cyclical encodings
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    
    # Clean Scheduled Times to Minutes
    df['CRSDepTime_Mins'] = (df['CRSDepTime'] // 100) * 60 + (df['CRSDepTime'] % 100)
    df['CRSArrTime_Mins'] = (df['CRSArrTime'] // 100) * 60 + (df['CRSArrTime'] % 100)
    
    df['DepHour'] = df['CRSDepTime'] // 100
    df['DepHour_sin'] = np.sin(2 * np.pi * df['DepHour'] / 24.0)
    df['DepHour_cos'] = np.cos(2 * np.pi * df['DepHour'] / 24.0)
    
    # 3. Holiday Flags (US Holidays 2022 or travel windows around them)
    print("Engineering holiday flags...")
    holidays_2022 = {
        '2022-01-01',  # New Year's Day
        '2022-01-17',  # MLK Day
        '2022-02-21',  # Presidents' Day
        '2022-05-30',  # Memorial Day
        '2022-06-19', '2022-06-20',  # Juneteenth
        '2022-07-04',  # Independence Day
        '2022-09-05',  # Labor Day
        '2022-10-10',  # Columbus Day
        '2022-11-11',  # Veterans Day
        '2022-11-23', '2022-11-24', '2022-11-25', '2022-11-26', '2022-11-27',  # Thanksgiving window
        '2022-12-23', '2022-12-24', '2022-12-25', '2022-12-26', '2022-12-27',  # Christmas window
        '2022-12-31'   # New Year's Eve
    }
    # Check membership
    df['Is_Holiday'] = df['FlightDate'].dt.strftime('%Y-%m-%d').isin(holidays_2022).astype(int)
    
    # 4. Hourly Airport Congestion Scores
    print("Engineering congestion metrics...")
    # Count departures per (Origin, Date, Hour)
    dep_counts = df.groupby(['Origin', 'FlightDate', 'DepHour']).size().reset_index(name='Origin_Dep_Congestion')
    df = pd.merge(df, dep_counts, on=['Origin', 'FlightDate', 'DepHour'], how='left')
    
    # Count arrivals per (Dest, Date, Hour)
    df['ArrHour'] = df['CRSArrTime'] // 100
    arr_counts = df.groupby(['Dest', 'FlightDate', 'ArrHour']).size().reset_index(name='Dest_Arr_Congestion')
    df = pd.merge(df, arr_counts, on=['Dest', 'FlightDate', 'ArrHour'], how='left')
    
    # Fill any NaNs with 0
    df['Origin_Dep_Congestion'] = df['Origin_Dep_Congestion'].fillna(0).astype(int)
    df['Dest_Arr_Congestion'] = df['Dest_Arr_Congestion'].fillna(0).astype(int)
    df.drop('ArrHour', axis=1, inplace=True)
    
    # 5. Historical Delay Risks (Avoiding data leakage by calculating overall statistics)
    print("Engineering historical delay risks...")
    df['Is_Delayed'] = (df['ArrDelay'] > 15).astype(int)
    
    airline_delay_prob = df.groupby('Operating_Airline')['Is_Delayed'].mean()
    origin_delay_prob = df.groupby('Origin')['Is_Delayed'].mean()
    dest_delay_prob = df.groupby('Dest')['Is_Delayed'].mean()
    
    df['Airline_Delay_Risk'] = df['Operating_Airline'].map(airline_delay_prob)
    df['Origin_Airport_Risk'] = df['Origin'].map(origin_delay_prob)
    df['Dest_Airport_Risk'] = df['Dest'].map(dest_delay_prob)
    
    # Fill potential missing values with the global mean
    global_mean_delay = df['Is_Delayed'].mean()
    df['Airline_Delay_Risk'] = df['Airline_Delay_Risk'].fillna(global_mean_delay)
    df['Origin_Airport_Risk'] = df['Origin_Airport_Risk'].fillna(global_mean_delay)
    df['Dest_Airport_Risk'] = df['Dest_Airport_Risk'].fillna(global_mean_delay)
    
    df.drop('Is_Delayed', axis=1, inplace=True)
    
    # 6. Operational metrics
    print("Engineering operational metrics...")
    df['Total_Taxi_Time'] = df['TaxiOut'] + df['TaxiIn']
    df['Ground_Time_Ratio'] = np.where(df['ActualElapsedTime'] > 0, df['Total_Taxi_Time'] / df['ActualElapsedTime'], 0.0)
    df['Time_Made_Up'] = df['DepDelay'] - df['ArrDelay']
    
    # 7. Dimensionality Reduction (PCA)
    print("Running PCA dimensionality reduction...")
    pca_features = [
        'Distance', 'CRSDepTime_Mins', 'CRSArrTime_Mins', 'DepDelay', 'ArrDelay', 
        'TaxiOut', 'TaxiIn', 'ActualElapsedTime', 'AirTime', 'Airline_Delay_Risk', 
        'Origin_Airport_Risk', 'Dest_Airport_Risk', 'Total_Taxi_Time', 
        'Ground_Time_Ratio', 'Time_Made_Up', 'Origin_Dep_Congestion', 'Dest_Arr_Congestion'
    ]
    
    # Extract only numeric PCA columns, filling any missing values with 0
    X_pca = df[pca_features].fillna(0).values
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    # Run 2D PCA
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    df['PCA_1'] = components[:, 0]
    df['PCA_2'] = components[:, 1]
    
    print(f"PCA completed. Variance explained: {round(pca.explained_variance_ratio_.sum() * 100, 2)}%")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # 8. Save output (all columns)
    print(f"Saving final dataset to {output_parquet}...")
    df.to_parquet(output_parquet, index=False, compression='snappy')
    print("Feature engineering successfully completed!")
    print(f"Processed shape: {df.shape}")

if __name__ == "__main__":
    run_feature_engineering()
