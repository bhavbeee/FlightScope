# FlightScope: Interactive Visual Analytics of US Flight Operations and Delays

## Overview
FlightScope is an interactive visual analytics system designed for exploring US domestic flight operations. Air transportation generates massive amounts of operational data, and analyzing this data provides valuable insights into airline performance, airport efficiency, and factors affecting flight reliability. By combining spatial, temporal, network, and high-dimensional analytics, FlightScope helps users discover trends, compare operational performance, and gain meaningful insights from large-scale aviation data.

---

## Data Source
The primary data source is the US Domestic Flight Operations Dataset (2022). 
*   **Origin:** Sourced from the Bureau of Transportation Statistics (BTS) and hosted on Kaggle.
*   **Scale:** Contains approximately 7 million flight records distributed across monthly CSV files.
*   **Dimensionality:** Features approximately 120 attributes, including temporal scheduling, airport metadata, and detailed delay metrics.

---

## Key Features & Visualizations
The application follows the visual information-seeking mantra: "Overview first, zoom and filter, then details-on-demand".
*   **Air Traffic Network Explorer:** Uses interactive network graphs and route maps to explore flight connections across the US.
*   **Airport Delay Heatmap:** Analyzes average delays and seasonal patterns using interactive temporal-spatial heatmaps.
*   **Airline Performance Dashboard:** Benchmarks airline efficiency based on delays and cancellation rates using bar, line, and radar charts.
*   **Delay Cause Analysis:** Uses an interactive Sankey diagram to represent the flow of delay minutes from categories like weather, carrier, security, and NAS delays.
*   **High-Dimensional Flight Analytics:** Employs Parallel Coordinates Plots and standardizes metrics using $X_{new}=\frac{x-\mu}{\sigma}$ alongside PCA, t-SNE, or UMAP projections.

---

## System Architecture
To deliver a premium, responsive experience without freezing the browser, the system architecture separates data processing from rendering using a high-performance 4-tier stack.
*   **Data Tier:** Apache Parquet and DuckDB for sub-50ms analytical queries.
*   **Framework Tier:** Python Dash, Flask, and Dash Bootstrap Components to fit all views on one screen.
*   **Visualization Tier:** Plotly Graph Objects utilizing WebGL for lag-free rendering, and NetworkX for route centrality.
*   **Analytics Tier:** Scikit-learn for standardization and dimensionality reduction.

---

## Project Team
This project was developed by Group 17 as part of the CS661: Big Data Visual Analytics course. 
*   Anushka Rajora
*   Aditi
*   Raparthi Bhavishitha
*   Boppudi Sai Chaitanya
*   Devesh Kumar
*   Jiya Agarwal
*   Utkarsh Singhal
*   Kanishka S