import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# =========================
# Paths
# =========================
CSV_PATH = "/home/maria/MouseBox/selected_data/models/gpu_working/video_preds/session_view.csv"

# Must exist under ./assets/video_frames
VIDEO_FRAMES_URL = "/assets/video_frames_kp"


# =========================
# Takens embedding
# =========================
def takens_embedding(X, delay=3, dimension=8):
    T, D = X.shape
    max_shift = (dimension - 1) * delay
    return np.hstack(
        [X[i * delay : T - max_shift + i * delay] for i in range(dimension)]
    )


# =========================
# Load pose data
# =========================
dat = pd.read_csv(CSV_PATH)

bodyparts = dat.iloc[0, 1:].values
coords = dat.iloc[1, 1:].values
columns = [f"{bp}_{c}" for bp, c in zip(bodyparts, coords)]

X = dat.iloc[2:, 1:].astype(float)
X.columns = columns
X.reset_index(drop=True, inplace=True)

xy_cols = [c for c in X.columns if c.endswith("_x") or c.endswith("_y")]
X_xy = X[xy_cols].values.reshape(-1, 3, 2)

# Keep absolute position
X_uncentered = X_xy.reshape(len(X_xy), -1)

# Smooth + velocity
X_smooth = savgol_filter(X_uncentered, 11, 2, axis=0)
V = np.diff(X_smooth, axis=0, prepend=X_smooth[[0]])
X_feat = np.concatenate([X_smooth, V], axis=1)


# =========================
# Takens + PCA
# =========================
tau = 3
m = 8

X_takens = takens_embedding(X_feat, delay=tau, dimension=m)
X_takens = StandardScaler().fit_transform(X_takens)
Z = PCA(n_components=3).fit_transform(X_takens)

T_video = len(X_feat)
T_takens = len(Z)
takens_offset = (m - 1) * tau


# =========================
# Figure factory
# =========================
def make_figure(video_t):
    t_takens = video_t - takens_offset
    t_takens = max(0, min(T_takens - 1, t_takens))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=Z[:, 0],
            y=Z[:, 1],
            z=Z[:, 2],
            mode="lines",
            line=dict(width=2),
            name="Takens trajectory",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[Z[t_takens, 0]],
            y=[Z[t_takens, 1]],
            z=[Z[t_takens, 2]],
            mode="markers",
            marker=dict(size=7, color="red"),
            name="Current state",
        )
    )

    fig.update_layout(
        title=f"Takens embedding (video frame {video_t})",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


# =========================
# Dash app
# =========================
app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(
            id="takens-plot",
            figure=make_figure(0),
            style={"height": "500px"},
        ),

        html.Div(
            [
                html.Button("▶ Play / Pause", id="play-button", n_clicks=0),
                dcc.Interval(
                    id="play-interval",
                    interval=100,   # ms (10 FPS)
                    disabled=True,
                ),
            ],
            style={"margin": "10px 0"},
        ),

        html.Div(id="frame-label", style={"marginBottom": "5px"}),

        dcc.Slider(
            id="time-slider",
            min=0,
            max=T_video - 1,
            step=1,
            value=0,
            updatemode="mouseup",
            marks=None,      # <-- removes cluttered numbers
        ),

        html.Img(
            id="video-frame",
            style={"width": "640px", "marginTop": "10px"},
        ),
    ]
)


# =========================
# Play / pause toggle
# =========================
@app.callback(
    Output("play-interval", "disabled"),
    Input("play-button", "n_clicks"),
    State("play-interval", "disabled"),
)
def toggle_play(n_clicks, disabled):
    return not disabled


# =========================
# Advance slider when playing
# =========================
@app.callback(
    Output("time-slider", "value"),
    Input("play-interval", "n_intervals"),
    State("time-slider", "value"),
)
def advance_slider(_, current_t):
    if current_t is None:
        return 0
    return min(current_t + 1, T_video - 1)


# =========================
# Main update callback
# =========================
@app.callback(
    Output("takens-plot", "figure"),
    Output("video-frame", "src"),
    Output("frame-label", "children"),
    Input("time-slider", "value"),
)
def update(video_t):
    fig = make_figure(video_t)
    img_src = f"{VIDEO_FRAMES_URL}/frame_{video_t:06d}.jpg"
    label = f"Video frame: {video_t} / {T_video - 1}"
    return fig, img_src, label


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
