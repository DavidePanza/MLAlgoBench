import streamlit as st
import plotly.graph_objects as go

def plot_results(results_df, metric):
    fig = go.Figure([go.Bar(x=results_df[metric], y=results_df["Model"], orientation='h')])
    fig.update_layout(
    title=f"{metric}",
            xaxis_title="Year",
            yaxis=dict(title="Models"),
            yaxis2=dict(
                title="Model",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.1, y=1.1)
            )
    st.plotly_chart(fig, use_container_width=True)