import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retentioniq import config
from retentioniq.data.loader import load_training_data
from ui import common
from ui.theme import SEQUENTIAL, SERIES, layout

common.header("Segments & cohorts",
              "Observed churn rates by cohort in the reference data, next to the model's average estimate.")
common.note("These are <b>associations</b> in historical data. A segment with high churn does not mean "
            "the attribute causes churn, and changing it would not necessarily retain anyone.", "warn")

min_n = config.settings()["segments"]["min_cohort_size"]


@st.cache_data(show_spinner=False)
def reference() -> pd.DataFrame:
    df, _ = load_training_data()
    df["Tenure band"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72, 200],
                               labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo", "72+ mo"])
    df["Monthly charges band"] = pd.cut(df["MonthlyCharges"], [0, 35, 70, 90, 1000],
                                        labels=["< $35", "$35-70", "$70-90", "$90+"])
    df["Add-on services"] = (df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]]
                             == "Yes").sum(axis=1).astype(str)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


df = reference()
latest = common.db().latest_scores()
dims = ["Contract", "Tenure band", "InternetService", "PaymentMethod", "Monthly charges band",
        "Add-on services", "TechSupport", "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]
dim = st.selectbox("Compare cohorts by", dims)

g = df.groupby(dim, observed=True)["Churn"].agg(["sum", "count"]).reset_index()
g.columns = [dim, "churners", "customers"]
g["churn_rate"] = g["churners"] / g["customers"]
ci = [wilson(int(k), int(n)) for k, n in zip(g["churners"], g["customers"], strict=True)]
g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
hidden = g[g["customers"] < min_n]
g = g[g["customers"] >= min_n]
if dim in ("Tenure band", "Monthly charges band", "Add-on services"):
    g = g.sort_values(dim)
else:
    g = g.sort_values("churn_rate", ascending=False)

if not latest.empty and dim in latest.columns:
    pred = latest.groupby(dim)["churn_probability"].mean().rename("avg_predicted")
    g = g.merge(pred, left_on=dim, right_index=True, how="left")

fig = go.Figure()
fig.add_bar(x=g[dim].astype(str), y=g["churn_rate"], name="Observed churn rate (reference data)",
            marker=dict(color=SERIES[0], cornerradius=4),
            error_y=dict(type="data", symmetric=False, array=g["ci_high"] - g["churn_rate"],
                         arrayminus=g["churn_rate"] - g["ci_low"], color="#64748B", thickness=1.2, width=4),
            customdata=g[["customers", "ci_low", "ci_high"]].values,
            hovertemplate="%{x}<br>Observed churn %{y:.1%} (95% CI %{customdata[1]:.1%}-%{customdata[2]:.1%})"
                          "<br>n = %{customdata[0]:,}<extra></extra>")
if "avg_predicted" in g:
    fig.add_scatter(x=g[dim].astype(str), y=g["avg_predicted"], mode="markers",
                    name="Avg. estimated risk (scored portfolio)",
                    marker=dict(color=SERIES[1], size=11, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                    hovertemplate="%{x}<br>Avg. estimated risk %{y:.1%}<extra></extra>")
layout(fig, height=380, legend=True, ytitle="Churn rate", xtitle=dim)
fig.update_yaxes(tickformat=".0%", rangemode="tozero")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption(f"Error bars: 95% Wilson interval. Cohorts with fewer than {min_n} customers are suppressed"
           + (f" ({len(hidden)} hidden)." if len(hidden) else "."))

st.dataframe(g.rename(columns={"churn_rate": "observed churn", "avg_predicted": "avg est. risk"}),
             hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in ["observed churn", "ci_low", "ci_high", "avg est. risk"]})

st.subheader("Contract × tenure")
pivot_n = df.pivot_table(index="Contract", columns="Tenure band", values="Churn", aggfunc="count", observed=True)
pivot = df.pivot_table(index="Contract", columns="Tenure band", values="Churn", aggfunc="mean", observed=True)
masked = pivot.where(pivot_n >= min_n)
text = np.where(masked.notna(), (masked * 100).round(0).astype("Int64").astype(str) + "%", "n<" + str(min_n))
fig = go.Figure(go.Heatmap(
    z=masked.values, x=[str(c) for c in masked.columns], y=masked.index,
    colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
    zmin=0, zmax=float(np.nanmax(masked.values)), xgap=2, ygap=2,
    text=text, texttemplate="%{text}", customdata=pivot_n.values,
    hovertemplate="%{y} · %{x}<br>Observed churn %{z:.1%}<br>n = %{customdata:,}<extra></extra>",
    colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=10)))
layout(fig, height=280)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
