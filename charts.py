"""Plotly figure builders. Pure functions: data in, styled figure out."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from oee import sort_shifts
from theme import (AMBER, BG, CYAN, FONT, GRID, GREEN, MINT, MONO, MUTED, RED,
                   SLATE, SURFACE, TEXT, TEXT_SOFT, VIOLET)

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def base_layout(height: int, title: str | None = None, **over) -> dict:
    lay = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT_SOFT, size=12),
        margin=dict(l=62, r=24, t=56 if title else 24, b=48),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=MINT, font=dict(family=MONO, size=12, color=TEXT)),
        title=dict(text=f"<b>{title}</b>" if title else None, x=0.01, xanchor="left",
                   font=dict(size=15, color=TEXT, family=FONT)),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(size=11, family=MONO),
                   title=dict(font=dict(size=12, color=MUTED))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(size=11, family=MONO),
                   title=dict(font=dict(size=12, color=MUTED))),
        showlegend=False,
    )
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(lay.get(k), dict):
            lay[k] = {**lay[k], **v}
        else:
            lay[k] = v
    return lay


def _limit_labels(fig, items):
    for y, text, color in items:
        fig.add_annotation(x=1, xref="paper", y=y, text=text, showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=color, size=11, family=MONO))


def _padded_range(values, *limits, pad=0.12):
    lo, hi = min(np.nanmin(values), *limits), max(np.nanmax(values), *limits)
    span = hi - lo or 1.0
    return [lo - pad * span, hi + pad * span]


# ─── Control charts ─────────────────────────────────────────────────────────
def control_chart(cc, x_idx, x_label, x_title):
    fig = go.Figure()
    s = cc.sigma_stat
    zones = [(-1, 1, GREEN, 0.045), (1, 2, AMBER, 0.05), (-2, -1, AMBER, 0.05),
             (2, 3, RED, 0.06), (-3, -2, RED, 0.06)]
    for lo, hi, col, op in zones:
        fig.add_hrect(y0=cc.cl + lo * s, y1=cc.cl + hi * s, fillcolor=col, opacity=op, line_width=0, layer="below")
    fig.add_vrect(x0=0.5, x1=cc.baseline + 0.5, fillcolor=VIOLET, opacity=0.08, line_width=0, layer="below")
    fig.add_annotation(x=0.5 + cc.baseline / 2, y=1, yref="paper", text="BASELINE · PHASE I", showarrow=False,
                       yanchor="bottom", font=dict(size=10, color=VIOLET, family=MONO))

    for y, col, dash, w in ((cc.cl, MINT, "solid", 2.2), (cc.ucl, RED, "dash", 1.8), (cc.lcl, RED, "dash", 1.8)):
        fig.add_hline(y=y, line=dict(color=col, width=w, dash=dash))
    _limit_labels(fig, [(cc.ucl, f"<b>UCL</b> {cc.ucl:.4f}", RED), (cc.cl, f"<b>CL</b> {cc.cl:.4f}", MINT),
                        (cc.lcl, f"<b>LCL</b> {cc.lcl:.4f}", RED)])

    fig.add_trace(go.Scatter(
        x=x_idx, y=cc.x, mode="lines+markers", name=x_label,
        line=dict(color=CYAN, width=1.8), marker=dict(size=5, color=CYAN, line=dict(width=1, color=BG)),
        hovertemplate="#%{x}<br>%{y:.4f}<extra></extra>"))
    flagged = sorted(cc.signals)
    if flagged:
        fx, fy = [x_idx[i] for i in flagged], [cc.x[i] for i in flagged]
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers", hoverinfo="skip",
                                 marker=dict(size=20, color=RED, opacity=0.18)))
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="markers", name="Signal",
            marker=dict(size=9, color=RED, line=dict(color="white", width=1.2)),
            text=[", ".join(cc.signals[i]) for i in flagged],
            hovertemplate="#%{x}<br>%{y:.4f}<br><b>Rules: %{text}</b><extra></extra>"))
    fig.update_layout(**base_layout(
        400, f"{cc.kind} chart · {x_label}", margin=dict(r=120),
        xaxis=dict(title=dict(text=x_title), range=[0.5, len(x_idx) + 0.5]),
        yaxis=dict(tickformat=".3f", range=_padded_range(cc.x, cc.ucl, cc.lcl))))
    return fig


def spread_chart(cc, x_idx, s_label):
    fig = go.Figure()
    fig.add_hline(y=cc.cl_spread, line=dict(color=MINT, width=2))
    fig.add_hline(y=cc.ucl_spread, line=dict(color=RED, width=1.8, dash="dash"))
    labels = [(cc.ucl_spread, f"<b>UCL</b> {cc.ucl_spread:.4f}", RED), (cc.cl_spread, f"<b>CL</b> {cc.cl_spread:.4f}", MINT)]
    if cc.lcl_spread > 0:
        fig.add_hline(y=cc.lcl_spread, line=dict(color=RED, width=1.8, dash="dash"))
        labels.append((cc.lcl_spread, f"<b>LCL</b> {cc.lcl_spread:.4f}", RED))
    _limit_labels(fig, labels)
    fig.add_trace(go.Scatter(
        x=x_idx, y=cc.spread, mode="lines+markers", line=dict(color=AMBER, width=1.7),
        marker=dict(size=5, color=AMBER, line=dict(width=1, color=BG)),
        hovertemplate="#%{x}<br>%{y:.4f}<extra></extra>"))
    if cc.spread_signals:
        fx, fy = [x_idx[i] for i in cc.spread_signals], [cc.spread[i] for i in cc.spread_signals]
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers", hoverinfo="skip",
                                 marker=dict(size=20, color=RED, opacity=0.18)))
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers",
                                 marker=dict(size=9, color=RED, line=dict(color="white", width=1.2)),
                                 hovertemplate="#%{x}<br>%{y:.4f}<extra>Range signal</extra>"))
    fig.update_layout(**base_layout(
        270, f"{s_label} chart · variation", margin=dict(r=120),
        xaxis=dict(range=[0.5, len(x_idx) + 0.5]),
        yaxis=dict(tickformat=".3f", rangemode="tozero")))
    return fig


# ─── Capability ─────────────────────────────────────────────────────────────
def _cap_color(v: float) -> str:
    return RED if v < 1 else AMBER if v < 1.33 else MINT if v < 1.67 else GREEN


def capability_gauges(cap):
    fig = go.Figure()
    items = [("Cp", cap.cp, "potential"), ("Cpk", cap.cpk, "short-term"),
             ("Pp", cap.pp, "overall"), ("Ppk", cap.ppk, "long-term")]
    for i, (name, val, sub) in enumerate(items):
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=val,
            number=dict(valueformat=".2f", font=dict(family=MONO, size=40, color=TEXT)),
            title=dict(text=f"<b>{name}</b><br><span style='font-size:11px;color:{MUTED}'>{sub}</span>",
                       font=dict(size=17, color=TEXT_SOFT)),
            domain=dict(x=[i / 4 + 0.015, (i + 1) / 4 - 0.015], y=[0, 1]),
            gauge=dict(
                axis=dict(range=[0, 2], tickvals=[1, 1.33, 1.67], ticktext=["1.0", "1.33", "1.67"],
                          tickfont=dict(size=10, color=MUTED, family=MONO), tickcolor=SLATE),
                bar=dict(color=_cap_color(val), thickness=0.32), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                steps=[dict(range=[0, 1], color="rgba(255,84,112,0.16)"),
                       dict(range=[1, 1.33], color="rgba(255,176,32,0.15)"),
                       dict(range=[1.33, 2], color="rgba(25,227,177,0.13)")],
                threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.82, value=1.33))))
    fig.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=70, b=10),
                      font=dict(family=FONT, color=TEXT_SOFT))
    return fig


def capability_histogram(values, cap, lsl, usl):
    v = np.asarray(values, float).ravel()
    pad = 2.5 * cap.sigma_overall
    x0, x1 = min(v.min(), lsl) - pad, max(v.max(), usl) + pad
    xs = np.linspace(x0, x1, 400)
    pdf = np.exp(-0.5 * ((xs - cap.mean) / cap.sigma_overall) ** 2) / (cap.sigma_overall * np.sqrt(2 * np.pi))
    fig = go.Figure()
    fig.add_vrect(x0=x0, x1=lsl, fillcolor=RED, opacity=0.13, line_width=0, layer="below")
    fig.add_vrect(x0=usl, x1=x1, fillcolor=RED, opacity=0.13, line_width=0, layer="below")
    fig.add_vrect(x0=lsl, x1=usl, fillcolor=MINT, opacity=0.035, line_width=0, layer="below")
    fig.add_trace(go.Histogram(x=v, histnorm="probability density", nbinsx=42, marker_color=CYAN, opacity=0.6,
                               marker_line=dict(color=BG, width=1), hovertemplate="%{x:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", line=dict(color=MINT, width=3), fill="tozeroy",
                             fillcolor="rgba(25,227,177,0.08)", hoverinfo="skip"))
    for val, name in ((lsl, "LSL"), (usl, "USL")):
        fig.add_vline(x=val, line=dict(color=RED, width=2, dash="dash"))
        fig.add_annotation(x=val, y=1, yref="paper", text=f"<b>{name}</b> {val:.3f}", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color=RED, family=MONO))
    fig.add_vline(x=cap.mean, line=dict(color=VIOLET, width=1.6, dash="dot"))
    fig.add_annotation(x=cap.mean, y=1, yref="paper", text=f"mean {cap.mean:.4f}", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=VIOLET, family=MONO))
    fig.update_layout(**base_layout(
        380, "Distribution vs specification", margin=dict(t=64),
        xaxis=dict(title=dict(text="Measurement"), range=[x0, x1], tickformat=".3f"),
        yaxis=dict(title=dict(text="Density"), showticklabels=False)))
    return fig


# ─── OEE ────────────────────────────────────────────────────────────────────
def oee_gauge(oee_value: float, benchmark: float = 0.85):
    pct = oee_value * 100
    color = RED if pct < 60 else AMBER if pct < benchmark * 100 else MINT
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", valueformat=".1f", font=dict(family=MONO, size=54, color=TEXT)),
        title=dict(text="<b>OEE</b><br><span style='font-size:12px;color:%s'>vs 85%% benchmark</span>" % MUTED,
                   font=dict(size=18, color=TEXT_SOFT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 25, 50, 60, 85, 100], tickfont=dict(size=10, color=MUTED, family=MONO),
                      tickcolor=SLATE),
            bar=dict(color=color, thickness=0.30), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 60], color="rgba(255,84,112,0.16)"),
                   dict(range=[60, benchmark * 100], color="rgba(255,176,32,0.15)"),
                   dict(range=[benchmark * 100, 100], color="rgba(25,227,177,0.14)")],
            threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.85, value=benchmark * 100))))
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=30, r=30, t=80, b=10),
                      font=dict(family=FONT, color=TEXT_SOFT))
    return fig


def waterfall(w):
    labels = ["Planned time", "Downtime", "Speed loss", "Quality loss", "Fully productive"]
    h = [w["planned"] / 60, w["downtime"] / 60, w["speed"] / 60, w["quality"] / 60, w["productive"] / 60]
    after1 = h[0] - h[1]
    after2 = after1 - h[2]
    after3 = after2 - h[3]
    bases = [0, after1, after2, after3, 0]
    texts = [f"{h[0]:,.0f} h", f"−{h[1]:,.0f} h", f"−{h[2]:,.0f} h", f"−{h[3]:,.0f} h", f"{h[4]:,.0f} h"]
    pct = [None] + [f"{v / h[0] * 100:.1f}% of planned" for v in h[1:4]] + [f"{h[4] / h[0] * 100:.1f}% of planned"]
    fig = go.Figure(go.Bar(
        x=labels, y=h, base=bases, width=0.62, marker=dict(color=[SLATE, RED, RED, RED, MINT]),
        text=texts, textposition="outside", textfont=dict(family=MONO, size=13, color=TEXT), cliponaxis=False,
        customdata=[p or "" for p in pct],
        hovertemplate="%{x}<br>%{y:,.0f} h<br>%{customdata}<extra></extra>"))
    levels = [h[0], after1, after2, after3]
    for i, y in enumerate(levels):
        fig.add_shape(type="line", x0=i + 0.31, x1=i + 0.69, y0=y, y1=y, xref="x", yref="y",
                      line=dict(color=SLATE, width=1.5, dash="dot"))
    fig.update_layout(**base_layout(
        360, "Where the planned time goes", margin=dict(t=64),
        yaxis=dict(title=dict(text="Hours"), tickformat=",.0f", range=[0, h[0] * 1.14])))
    return fig


def pareto(p):
    colors, seen80 = [], False
    for c in p["cum_pct"]:
        colors.append(SLATE if seen80 else AMBER)
        if c >= 80:
            seen80 = True
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p["reason"], y=p["minutes"] / 60, marker=dict(color=colors, line=dict(width=0)),
                         text=[f"{m / 60:,.0f} h" for m in p["minutes"]], textposition="outside",
                         textfont=dict(family=MONO, size=12, color=TEXT), hovertemplate="%{x}<br>%{y:,.0f} h<extra></extra>"))
    fig.add_trace(go.Scatter(x=p["reason"], y=p["cum_pct"], yaxis="y2", mode="lines+markers",
                             line=dict(color=MINT, width=3), marker=dict(size=8, color=MINT, line=dict(color=BG, width=2)),
                             hovertemplate="%{x}<br>%{y:.0f}% cumulative<extra></extra>"))
    fig.add_hline(y=80, yref="y2", line=dict(color=MUTED, dash="dot", width=1.2))
    fig.update_layout(**base_layout(
        360, "Downtime Pareto", margin=dict(r=48, t=64),
        yaxis=dict(title=dict(text="Hours"), range=[0, float(p["minutes"].max() / 60) * 1.2]),
        yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False,
                    tickfont=dict(size=11, family=MONO), linecolor=GRID)))
    return fig


def daily_trend(dd, benchmark: float = 0.85):
    d = dd.copy()
    d["ma"] = d["oee"].rolling(7, min_periods=3).mean()
    fig = go.Figure()
    fig.add_hline(y=benchmark * 100, line=dict(color=MINT, dash="dash", width=1.6))
    fig.add_annotation(x=1, xref="paper", y=benchmark * 100, text=f"<b>{benchmark * 100:.0f}%</b> benchmark",
                       showarrow=False, xanchor="left", xshift=8, font=dict(color=MINT, size=11, family=MONO))
    fig.add_trace(go.Scatter(x=d["date"], y=d["oee"] * 100, mode="markers", name="Daily",
                             marker=dict(size=6, color=CYAN, opacity=0.45),
                             hovertemplate="%{x}<br>%{y:.1f}%<extra>Daily</extra>"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ma"] * 100, mode="lines", name="7-day average",
                             line=dict(color=CYAN, width=3.2, shape="spline", smoothing=0.7), fill="tozeroy",
                             fillcolor="rgba(76,201,240,0.07)",
                             hovertemplate="%{x}<br>%{y:.1f}%<extra>7-day avg</extra>"))
    fig.update_layout(**base_layout(
        360, "Daily OEE and 7-day trend", margin=dict(r=120, t=64),
        yaxis=dict(range=[40, 100], ticksuffix="%")))
    return fig


def shift_bars(bs):
    fig = go.Figure()
    order = sort_shifts(bs["shift"].tolist())
    spec = [("availability", "Availability", CYAN), ("performance", "Performance", VIOLET),
            ("quality", "Quality", GREEN), ("oee", "OEE", AMBER)]
    for key, name, col in spec:
        fig.add_trace(go.Bar(x=bs["shift"], y=bs[key] * 100, name=name, marker=dict(color=col),
                             text=[f"{v * 100:.0f}" for v in bs[key]], textposition="outside",
                             textfont=dict(family=MONO, size=11, color=TEXT),
                             hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
    fig.update_layout(**base_layout(
        360, "OEE components by shift", barmode="group", bargap=0.28, bargroupgap=0.06, margin=dict(t=64),
        showlegend=True, legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=11)),
        xaxis=dict(categoryorder="array", categoryarray=order),
        yaxis=dict(range=[0, 112], ticksuffix="%")))
    return fig
