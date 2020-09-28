# @author lucasmiranda42

# Add to the system path for the script to find deepof
import sys

sys.path.insert(1, "../")

from sys import argv
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

script, samples, df1, df2, df3, out = argv
samples = int(samples)

# Load plotting data
dfencs = pd.read_hdf(df1)
clust_occur = pd.read_hdf(df2)
maedf = pd.read_hdf(df3)

# Build figures
fig = px.scatter(
    data_frame=dfencs.sort_values(["epoch", "cluster"]),
    x="x",
    y="y",
    animation_frame="epoch",
    color="cluster",
    labels={"x": "LDA 1", "y": "LDA 2"},
    width=550,
    height=500,
    color_discrete_sequence=px.colors.qualitative.T10,
)

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
fig.update_xaxes(showgrid=True, range=[-0.2, 1.2])
fig.update_yaxes(showgrid=True, range=[-0.2, 1.2])

fig.update_layout({"paper_bgcolor": "rgba(0, 0, 0, 0)"})

# Cluster membership animated over epochs
clust_occur = pd.DataFrame(
    dfencs.loc[:, ["cluster", "epoch"]].groupby(["epoch", "cluster"]).cluster.count()
)
clust_occur.rename(columns={"cluster": "count"}, inplace=True)
clust_occur = clust_occur.reset_index()
fig2 = px.bar(
    data_frame=clust_occur,
    x="cluster",
    y="count",
    animation_frame="epoch",
    color="cluster",
    labels={"cluster": "Cluster number", "count": "Count"},
    width=550,
    height=500,
    color_discrete_sequence=px.colors.qualitative.T10,
)

fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
fig2.update_xaxes(showgrid=True)
fig2.update_yaxes(showgrid=True, range=[0, samples])

fig2.update_layout({"paper_bgcolor": "rgba(0, 0, 0, 0)"})


# Scatterplot reconstruction over epochs
fig3 = px.scatter(
    data_frame=dfencs,
    x="trajectories",
    y="reconstructions",
    animation_frame="epoch",
    labels={"x": "original data", "y": "gmvae reconstruction"},
    width=550,
    height=500,
    color_discrete_sequence=px.colors.qualitative.T10,
)

fig3.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
fig3.update_xaxes(showgrid=True, range=[-2, 2])
fig3.update_yaxes(showgrid=True, range=[-2, 2])

# Histogram with confidence in best cluster
fig4 = px.histogram(
    data_frame=dfencs,
    x="confidence",
    animation_frame="epoch",
    labels={"confidence": "Confidence in best cluster"},
    width=550,
    height=500,
    color_discrete_sequence=px.colors.qualitative.T10,
)

fig4.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
fig4.update_xaxes(showgrid=True, range=[-0.05, 1.05])

fig4.update_layout({"paper_bgcolor": "rgba(0, 0, 0, 0)"})

fig3.update_layout({"paper_bgcolor": "rgba(0, 0, 0, 0)"})

fig5 = px.line(
    data_frame=maedf,
    x="epoch",
    y="mae",
    color_discrete_sequence=px.colors.qualitative.T10,
)
# fig5.update_xaxes(showgrid=False)
# fig5.update_yaxes(showgrid=False)

fig5.update_layout({"paper_bgcolor": "rgba(0, 0, 0, 0)"})

# Save figures as html
fig.write_html("{}_latent_LDA.html".format(out))
fig2.write_html("{}_cluster_distribution.html".format(out))
fig3.write_html("{}_scatter_reconstruction.html".format(out))
fig4.write_html("{}_confidence_histogram.html".format(out))

# Combine all three figures in a Dash application
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(children="deepof GMVAE training demo", className="row"),
        html.Div(
            [
                html.Div([dcc.Graph(figure=fig)], className="six columns"),
                html.Div([dcc.Graph(figure=fig2)], className="six columns"),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div([dcc.Graph(figure=fig3)], className="six columns"),
                html.Div([dcc.Graph(figure=fig4)], className="six columns"),
            ],
            className="row",
        ),
        html.Div(
            [html.Div([dcc.Graph(figure=fig5)], className="twelve columns")],
            className="row",
        ),
    ]
)

app.run_server(debug=True, use_reloader=False)
