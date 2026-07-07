import os
import glob
import pandas as pd
import numpy as np

# 66 columns to keep
columns_to_keep = [
    'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Marketing_Airline_Network',
    'Operated_or_Branded_Code_Share_Partners', 'DOT_ID_Marketing_Airline',
    'IATA_Code_Marketing_Airline', 'Flight_Number_Marketing_Airline',
    'Operating_Airline ', 'DOT_ID_Operating_Airline',
    'IATA_Code_Operating_Airline', 'Tail_Number',
    'Flight_Number_Operating_Airline', 'OriginAirportID', 'OriginAirportSeqID',
    'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState',
    'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportID',
    'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState',
    'DestStateFips', 'DestStateName', 'DestWac', 'CRSDepTime', 'DepTime',
    'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups',
    'DepTimeBlk', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime',
    'ArrTime', 'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups',
    'ArrTimeBlk', 'Cancelled', 'CancellationCode', 'Diverted', 'CRSElapsedTime',
    'ActualElapsedTime', 'AirTime', 'Flights', 'Distance', 'DistanceGroup',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
    'LateAircraftDelay', 'DivAirportLandings', 'Duplicate'
]

def clean_and_sample(filepath, target_samples=150000):
    print(f"Processing and cleaning {filepath}...")
    
    # 1. Drop completely (or almost) missing columns (accomplished via usecols)
    df = pd.read_csv(filepath, usecols=lambda c: c in columns_to_keep, low_memory=False)
    
    # Drop exact duplicates
    df.drop_duplicates(inplace=True)
    
    # 2. Add / Impute missing data
    # Fill delay columns with 0 (since BTS logs them as NaN when there is no delay)
    delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    existing_delays = [c for c in delay_cols if c in df.columns]
    if existing_delays:
        df[existing_delays] = df[existing_delays].fillna(0)
        
    # Fill CancellationCode
    if 'CancellationCode' in df.columns:
        df['CancellationCode'] = df['CancellationCode'].fillna('N/A')
        
    # Fill missing Tail_Number
    if 'Tail_Number' in df.columns:
        df['Tail_Number'] = df['Tail_Number'].fillna('UNKNOWN')
        
    # Filter out invalid operational rows (flights that are active but have missing times)
    if 'Cancelled' in df.columns and 'Diverted' in df.columns:
        critical_cols = ['DepTime', 'ArrTime', 'ActualElapsedTime']
        existing_crit = [col for col in critical_cols if col in df.columns]
        is_cancelled_or_diverted = (df['Cancelled'] == 1) | (df['Diverted'] == 1)
        has_valid_times = df[existing_crit].notnull().all(axis=1)
        df = df[is_cancelled_or_diverted | has_valid_times]
        
    # 3. Consistency of Datatypes (Prepare types before merging)
    # Ensure binary fields are boolean
    if 'Cancelled' in df.columns:
        df['Cancelled'] = df['Cancelled'].astype(bool)
    if 'Diverted' in df.columns:
        df['Diverted'] = df['Diverted'].astype(bool)
        
    # Standardize string fields (remove whitespace)
    string_cols = ['Marketing_Airline_Network', 'Operating_Airline', 'Origin', 'Dest', 'Tail_Number', 'CancellationCode']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    if 'FlightDate' in df.columns:
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
        
    # Stratify by Carrier and Delay status to keep rich representation of delays
    df['IsDelayedOrCancelled'] = (df['DepDelay'] > 0) | (df['ArrDelay'] > 0) | (df['Cancelled'] == True)
    
    grouped = df.groupby(['Marketing_Airline_Network', 'IsDelayedOrCancelled'], observed=True)
    sampled_chunks = []
    total_len = len(df)
    
    for _, group in grouped:
        group_n = int(round((len(group) / total_len) * target_samples))
        group_n = max(min(group_n, len(group)), min(150, len(group)))
        sampled_chunks.append(group.sample(n=group_n, random_state=42))
        
    sampled_df = pd.concat(sampled_chunks)
    
    if len(sampled_df) > target_samples:
        sampled_df = sampled_df.sample(n=target_samples, random_state=42)
    elif len(sampled_df) < target_samples:
        remaining = df.drop(sampled_df.index)
        needed = target_samples - len(sampled_df)
        sampled_df = pd.concat([sampled_df, remaining.sample(n=needed, random_state=42)])
        
    sampled_df.drop('IsDelayedOrCancelled', axis=1, inplace=True)
    return sampled_df

if __name__ == "__main__":
    csv_files = sorted(glob.glob("Flights_2022_*.csv"), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    all_sampled = []

    for filepath in csv_files:
        sampled_month = clean_and_sample(filepath, target_samples=150000)
        all_sampled.append(sampled_month)

    print("Combining all months...")
    final_df = pd.concat(all_sampled, ignore_index=True)
    
    # 4. Find and remove Outliers (After Merging)
    print("Finding and handling outliers...")
    
    # A. Physical impossibilities (e.g. AirTime <= 0 for non-cancelled/non-diverted flights)
    active_flights = (final_df['Cancelled'] == False) & (final_df['Diverted'] == False)
    
    invalid_rows = active_flights & (
        (final_df['AirTime'] <= 0) | 
        (final_df['TaxiOut'] <= 0) | 
        (final_df['TaxiIn'] <= 0) | 
        (final_df['ActualElapsedTime'] <= 0)
    )
    print(f"Removing {invalid_rows.sum()} rows with physically impossible values (AirTime/TaxiOut/TaxiIn <= 0)...")
    final_df = final_df[~invalid_rows]
    
    # B. Speed validation (Average flight speed must be reasonable for commercial jets)
    # Speed = Distance / (AirTime / 60)
    # Normal jet speeds are 400-600 mph. We filter out errors where speed > 750 mph or speed < 60 mph for long distances (>100 miles)
    active_flights = (final_df['Cancelled'] == False) & (final_df['Diverted'] == False)
    speed_mph = final_df['Distance'] / (final_df['AirTime'] / 60.0)
    impossible_speed = active_flights & (final_df['Distance'] > 100) & ((speed_mph > 750) | (speed_mph < 60))
    print(f"Removing {impossible_speed.sum()} rows with impossible flight speeds (>750 mph or <60 mph)...")
    final_df = final_df[~impossible_speed]
    
    # C. Statistical Outliers (Z-score > 5 for operational durations like TaxiOut and TaxiIn)
    # We use Z-score > 5 to catch extreme logging errors (e.g. taxi times of 10+ hours) while preserving real severe delays.
    for col in ['TaxiOut', 'TaxiIn', 'ActualElapsedTime']:
        col_mean = final_df[col].mean()
        col_std = final_df[col].std()
        # Find Z-score
        z_scores = (final_df[col] - col_mean) / col_std
        outliers = active_flights & (z_scores.abs() > 5)
        print(f"Removing {outliers.sum()} statistical outliers (Z-score > 5) from column '{col}'...")
        final_df = final_df[~outliers]

    # Optimize data types to save RAM
    for col in final_df.select_dtypes(include=['float64']).columns:
        final_df[col] = pd.to_numeric(final_df[col], downcast='float')
    for col in final_df.select_dtypes(include=['int64']).columns:
        final_df[col] = pd.to_numeric(final_df[col], downcast='integer')
        
    cat_cols = ['Marketing_Airline_Network', 'Operating_Airline', 'Origin', 'Dest', 'Tail_Number', 'CancellationCode']
    for col in cat_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('category')

    print(f"Final cleaned & merged dataset shape: {final_df.shape}")

    # Save outputs
    csv_out = "Flights_2022_sampled_1.8M.csv"
    pq_out = "Flights_2022_sampled_1.8M.parquet"
    
    print(f"Saving merged data to {csv_out}...")
    final_df.to_csv(csv_out, index=False)
    
    print(f"Saving merged data to {pq_out}...")
    final_df.to_parquet(pq_out, compression='snappy')
    print("All tasks completed successfully!")
