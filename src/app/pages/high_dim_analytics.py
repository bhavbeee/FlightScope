# 4.5 High-Dimensional Flight Analytics
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import duckdb
from src.pipeline.config import DB_PATH

def get_pca_data():
    """Fetches a random sample of 5000 flights for smooth rendering in the browser."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    # Changed Congestion_Score to Origin_Dep_Congestion to match your team's exact database schema
    query = """
        SELECT PCA_1, PCA_2, Origin_Dep_Congestion, Operating_Airline, ArrDelay 
        FROM flights 
        WHERE PCA_1 IS NOT NULL AND PCA_2 IS NOT NULL
        USING SAMPLE 5000
    """
    df = conn.execute(query).df()
    conn.close()
    return df

def create_layout():
    """Generates the UI layout for the High-Dimensional Analytics tab."""
    df = get_pca_data()
    
    # Build the PCA Scatter Plot using the correct column name
    fig = px.scatter(
        df, 
        x='PCA_1', 
        y='PCA_2', 
        color='Origin_Dep_Congestion',
        hover_data=['Operating_Airline', 'ArrDelay'],
        title="Flight Delay Clusters (Principal Component Analysis)",
        color_continuous_scale="Plasma",
        labels={'Origin_Dep_Congestion': 'Airport Congestion Score'},
        opacity=0.7
    )
    
    # Make it look sleek and modern
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,240,240,0.5)"
    )

    # Return the Dash Bootstrap layout
    return dbc.Container([
        html.H3("High-Dimensional Analytics", className="mt-4 mb-3 text-primary"),
        html.P("Visualizing dimensional reduction (PCA) to identify hidden clusters in flight delays and systemic congestion.", className="text-muted"),
        
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, id="pca-scatter-plot", style={"height": "65vh"})
            ])
        ], className="shadow-sm border-0")
    ], fluid=True)

# Expose the layout variable for app.py to import
layout = create_layout()