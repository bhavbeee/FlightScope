import os
import duckdb

def setup_database():
    parquet_path = "data/processed_flights.parquet"
    db_path = "data/flights.db"
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Processed parquet file not found at {parquet_path}. Please run feature engineering first.")
        
    print(f"Initializing DuckDB database at {db_path}...")
    
    # Connect to DuckDB (creates the file if it doesn't exist)
    conn = duckdb.connect(db_path)
    
    # Load parquet data into a table
    print("Loading data from parquet into 'flights' table...")
    conn.execute(f"CREATE OR REPLACE TABLE flights AS SELECT * FROM read_parquet('{parquet_path}')")
    
    # Create indexes for fast filtering in the dashboard
    print("Creating indexes on search fields...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_origin ON flights(Origin)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dest ON flights(Dest)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_carrier ON flights(Operating_Airline)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_month ON flights(Month)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON flights(FlightDate)")
    
    # Verify the table size
    row_count = conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    print(f"Successfully created 'flights' table with {row_count} rows!")
    
    # Check structure
    print("Database schema preview:")
    cols = conn.execute("PRAGMA table_info(flights)").fetchall()
    for col in cols[:10]:
        print(f" - {col[1]}: {col[2]}")
    print(f" ... and {len(cols) - 10} more columns.")
    
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    setup_database()
