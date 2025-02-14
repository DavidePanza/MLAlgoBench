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

        fig.update_layout(
            title_text=f"{feature}",
            title_x=0.5,  # Centers the title
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
    # Calculate correlation matrix
    corr = df.corr()

    # Create heatmap with correlation numbers on each cell
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        text=np.around(corr.values, decimals=2),   # Round correlation values
        texttemplate="%{text}",                     # Display text in each cell
        colorscale="Viridis",                       # Use a purple-ish colorscale
        zmin=zmin,                                    # Set the minimum correlation value for contrast
        zmax=zmax,                                     # Set the maximum correlation value for contrast
        colorbar=dict(title="Correlation")
    ))

    # Update layout
    fig.update_layout(
        title_text="Correlation Matrix",
        title_x=0.4,  # Centers the title (adjust if needed)
        title_font=dict(size=24),
        template="plotly_white"
    )
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)

def plot_results(results_df, metric, x_limits=None):
    """
    Plots a horizontal bar chart with improved visibility.
    
    Args:
    - results_df (pd.DataFrame): DataFrame containing model names and metric values.
    - metric (str): Column name representing the metric to be plotted.
    - x_limits (list, optional): A list containing [min, max] values for the x-axis range.
    """
    
    # Drop NaN values to prevent errors
    results_df = results_df.dropna(subset=[metric])

    # Set default x-axis limits
    if x_limits is None:
        if metric in ['R2 Score', 'Adjusted R2']:
            x_limits = [0, 1]  # Default for R² metrics
        else:
            x_limits = [results_df[metric].min() * 0.9, results_df[metric].max() * 1.1]  # Auto-scale

    # Create figure with enhanced bar styling
    fig = go.Figure([
        go.Bar(
            x=results_df[metric], 
            y=results_df["Model"], 
            orientation='h',
            marker=dict(
                color='royalblue',  # Customize bar color
                line=dict(color='white', width=1)  # Enhancing border thickness
            )
        )
    ])

    # Update layout with better spacing and label visibility
    fig.update_layout(
        title=dict(text=f"{metric} Comparison", font=dict(size=20)),  # Larger title font
        xaxis=dict(
            title=dict(text=metric, font=dict(size=16)),  # Bigger x-axis title
            tickfont=dict(size=14),  # Increase x-axis tick size
            showgrid=True, # Enable grid lines
            gridcolor="lightgray"  # Light gray gridlines
        ),
        yaxis=dict(
            title=dict(text="Models", font=dict(size=16)),  # Bigger y-axis title
            tickfont=dict(size=14),  # Increase y-axis tick size
            autorange="reversed"  # Keeps best models on top
        ),
        bargap=0.3,  # Space between bars
        margin=dict(l=120, r=20, t=50, b=50),
        template="plotly_white"
    )

    # Apply x-axis limits
    fig.update_xaxes(range=x_limits)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)



def plot_results(results_df, metric, x_limits=None):
    """
    Plots a horizontal bar chart with enhanced visibility and best models on top.
    
    Args:
    - results_df (pd.DataFrame): DataFrame containing model names and metric values.
    - metric (str): Column name representing the metric to be plotted.
    - x_limits (list, optional): A list containing [min, max] values for the x-axis range.
    """
    
    # Drop NaN values to prevent issues
    results_df = results_df.dropna(subset=[metric])

    # Sort models by metric (best on top)
    results_df.sort_values(by=metric, ascending=False, inplace=True)  # Ascending for horizontal bars

    # Set default x-axis limits
    if x_limits is None:
        if metric in ['R2 Score', 'Adjusted R2']:
            x_limits = [0, 1]  # Default for R² metrics
        else:
            x_limits = [results_df[metric].min() * 0.9, results_df[metric].max() * 1.1]  # Auto-scale

    # Create figure with enhanced styling
    fig = go.Figure([
        go.Bar(
            x=results_df[metric], 
            y=results_df["Model"], 
            orientation='h',
            marker=dict(
                color='royalblue',  # Customize bar color
                line=dict(color='black', width=0)  # Enhancing border thickness
            )
        )
    ])

    # Update layout with better spacing, label visibility, and improved grid
    fig.update_layout(
        #title=dict(text=f"{metric} Comparison", font=dict(size=20)),  # Larger title font
        xaxis=dict(
            title=dict(text=metric, font=dict(size=22)),  # Bigger x-axis title
            tickfont=dict(size=14),  # Increase x-axis tick size
            showgrid=True,  # Enable grid lines
            gridcolor="lightgray",  # Lighter grid color
            gridwidth=1,  # Thin grid lines
            griddash="dot"  # Make the grid lines dotted
        ),
        yaxis=dict(
            title=dict(text="Models", font=dict(size=26)),  # Bigger y-axis title
            tickfont=dict(size=14),  # Increase y-axis tick size
            autorange="reversed"  # Keeps best models on top
        ),
        bargap=0.3,  # Space between bars
        margin=dict(l=120, r=20, t=50, b=50),
        template="plotly_white"
    )

    # Apply x-axis limits
    fig.update_xaxes(range=x_limits)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)