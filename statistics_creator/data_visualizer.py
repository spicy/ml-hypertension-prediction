import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict
from constants import *

def plot_missing_data(missing_percentage_sorted: pd.Series, statistics_folder: str, column_definitions: Dict[str, str]):
    print("Plotting missing data...")
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_percentage_sorted.index,
        y=missing_percentage_sorted.values,
        text=missing_percentage_sorted.values.round(1).astype(str) + '%',
        textposition='outside',
        hovertext=[f"{col}<br>{column_definitions.get(col, 'No definition available')}<br>{val:.1f}%"
                   for col, val in zip(missing_percentage_sorted.index, missing_percentage_sorted.values)],
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Percentage of Missing Data by Column',
        xaxis_title='Columns',
        yaxis_title='Percentage of Missing Data',
        width=FIGURE_WIDTH * 50,
        height=FIGURE_HEIGHT * 50,
        xaxis_tickangle=-90,
        yaxis_range=[0, max(missing_percentage_sorted) * YLIM_MULTIPLIER]
    )

    # Save as interactive HTML
    html_path = os.path.join(statistics_folder, 'missing_data_percentage_interactive.html')
    pio.write_html(fig, file=html_path)
    print(f"Interactive missing data plot saved to: {html_path}")

    # Save as static image
    png_path = os.path.join(statistics_folder, 'missing_data_percentage.png')
    fig.write_image(png_path, scale=2)
    print(f"Static missing data plot saved to: {png_path}")

def plot_correlation_matrix(correlation_matrix: pd.DataFrame, statistics_folder: str, column_definitions: Dict[str, str]):
    print("Plotting correlation matrix...")
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    # Add hover text with full question definitions
    hover_text = [[f"{col}<br>{column_definitions.get(col, 'No definition available')}" for col in correlation_matrix.columns]
                  for _ in correlation_matrix.columns]

    fig.update_traces(
        hovertext=hover_text,
        hoverinfo='text'
    )

    fig.update_layout(
        title='Correlation Matrix Heatmap',
        width=CORRELATION_FIGURE_WIDTH * 50,
        height=CORRELATION_FIGURE_HEIGHT * 50,
        xaxis_title='',
        yaxis_title='',
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left', 'autorange': 'reversed'},
    )

    # Save as interactive HTML
    html_path = os.path.join(statistics_folder, 'correlation_matrix_heatmap_interactive.html')
    pio.write_html(fig, file=html_path)
    print(f"Interactive correlation matrix saved to: {html_path}")

    # Save as static image
    png_path = os.path.join(statistics_folder, 'correlation_matrix_heatmap.png')
    fig.write_image(png_path, scale=2)
    print(f"Static correlation matrix saved to: {png_path}")