import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_histograms(df, cols_x_row=4, bins=20):
    """
    Visualizes the distribution of each feature in a DataFrame using histograms.
    """
    cols = st.columns(cols_x_row)
    
    for idx, feature in enumerate(df.columns):
        # Create histogram for the current feature
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[feature], nbinsx=bins))
        
        # Set x-axis label only for the first plot in each row
        if idx % cols_x_row == 0:
            y_label = feature
        else:
            y_label = "" 

        fig.update_layout(yaxis=dict(
            tickfont=dict(size=14),  
            showgrid=True,  
            gridcolor="lightgray",  
            gridwidth=0,  
            griddash="dot"  
        ),  
            xaxis=dict(
            tickfont=dict(size=14)
        ),
            title_text=f"{feature}",
            title_x=0.3,  
            title_font=dict(size=24), 
            yaxis_title_text=y_label,
            bargap=0.05,  
            template="plotly_white"
        )
        
        # Use modulus to determine column index in the current row
        cols[idx % cols_x_row].plotly_chart(fig, use_container_width=True)
        
        # Every 4 histograms, create a new row of columns (except after the last iteration)
        if (idx + 1) % cols_x_row == 0 and (idx + 1) < len(df.columns):
            cols = st.columns(cols_x_row)

def plot_correlation_matrix(df,zmin=-1,zmax=1):
    """
    Visualizes the correlation matrix of a DataFrame using a heatmap.
    """
    corr = df.corr()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        text=np.around(corr.values, decimals=2),   
        texttemplate="%{text}",                     
        colorscale="Viridis",                      
        zmin=zmin,                                    
        zmax=zmax,                                    
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title_text="Correlation Matrix",
        title_x=0.4,  
        title_font=dict(size=24),
        template="plotly_white",
        height=600,
    )
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)

def plot_results(results_df, metric, x_limits=None):
    """
    Plots a horizontal bar chart displaying models' results metrics.
    The bset performing model is displayed at the top.
    """
    results_df = results_df.dropna(subset=[metric])

    # Sort models by metric (best on top)
    results_df.sort_values(by=metric, ascending=False, inplace=True)  

    # Set default x-axis limits according to the different metrics
    if x_limits is None:
        if metric in ['R2 Score', 'Adjusted R2']:
            x_limits = [0, 1]  
        else:
            x_limits = [results_df[metric].min() * 0.9, results_df[metric].max() * 1.1] 

    fig = go.Figure([
        go.Bar(
            x=results_df[metric], 
            y=results_df["Model"], 
            orientation='h',
            marker=dict(
                color='royalblue',  
                line=dict(color='black', width=0) 
            )
        )
    ])

    fig.update_layout(
        xaxis=dict(
            title=dict(text=metric, font=dict(size=22)),  
            tickfont=dict(size=14),  
            showgrid=True, 
            gridcolor="lightgray",  
            gridwidth=1,  
            griddash="dot"  
        ),
        yaxis=dict(
            title=dict(text="Models", font=dict(size=26)),  
            tickfont=dict(size=14),  
            autorange="reversed"  
        ),
        bargap=0.3,  
        margin=dict(l=120, r=20, t=50, b=50),
        template="plotly_white"
    )

    fig.update_xaxes(range=x_limits)
    st.plotly_chart(fig, use_container_width=True)