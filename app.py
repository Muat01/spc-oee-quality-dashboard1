"""
SPC & OEE Quality Dashboard

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
import oee
import spc
import ui

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="SPC & OEE Quality Dashboard", page_icon="📊", layout="wide")
ui.inject()


def plot(fig) -> None:
    """Render a Plotly figure inside a bordered 'card'."""
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config=charts.CONFIG)


# ─── Data loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def sample_spc() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_spc.csv")


@st.cache_data(show_spinner=False)
def sample_oee():
    return pd.read_csv(DATA / "sample_production_log.csv"), pd.read_csv(DATA / "sample_downtime_log.csv")


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    ui.sidebar_brand()
    st.markdown("**Data source**")
    source = st.radio("Source", ["Sample data (simulated)", "Upload my CSV files"], label_visibility="collapsed")
    spc_df, prod_df, down_df = sample_spc(), *sample_oee()
    if source.startswith("Upload"):
        up_spc = st.file_uploader("SPC data", type=["csv"], key="u1", help="Columns: subgroup, measurement")
        up_prod = st.file_uploader("OEE production log", type=["csv"], key="u2")
        up_down = st.file_uploader("OEE downtime log", type=["csv"], key="u3")
        with st.expander("Required columns"):
            st.code("SPC: subgroup, measurement\n"
                    "Production: date, shift, planned_min,\n  ideal_cycle_min, total_count, good_count\n"
                    "Downtime: date, shift, reason, minutes", language="text")
        if up_spc is not None:
            spc_df = pd.read_csv(up_spc)
        if up_prod is not None and up_down is not None:
            prod_df, down_df = pd.read_csv(up_prod), pd.read_csv(up_down)
        st.markdown('<div class="side-note">Any file left empty keeps the sample data for that tab.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-note">Simulated shaft-diameter process (25.000 ± 0.040 mm) and a '
                    'machining cell over 30 days × 3 shifts. Not real factory data.</div>', unsafe_allow_html=True)

# ─── Hero ───────────────────────────────────────────────────────────────────
ui.hero("Manufacturing analytics", "SPC &amp; OEE Quality Dashboard",
        "Control charts, process capability and equipment effectiveness in one place: "
        "see when a process changes, whether it meets specification, and where productive time is lost.",
        [("Simulated data" if source.startswith("Sample") else "Your data", "amber" if source.startswith("Sample") else "mint"),
         ("X̄-R · I-MR charts", "cyan"), ("Cp · Cpk · Pp · Ppk", "violet"), ("OEE loss analysis", "mint")])

tab_spc, tab_cap, tab_oee, tab_method = st.tabs(["📈  Control charts", "🎯  Capability", "⚙️  OEE", "ℹ️  Method"])

spc_error = None
try:
    ids, matrix = spc.to_subgroup_matrix(spc_df)
except Exception as exc:  # noqa: BLE001
    spc_error, ids, matrix = str(exc), [], np.empty((0, 0))

# ═══════════════════════════════ CONTROL CHARTS ═════════════════════════════
with tab_spc:
    if spc_error:
        st.error(spc_error)
    else:
        k, n = matrix.shape
        ui.section("Statistical process control", "Is the process stable?",
                   "Limits come from a baseline window (Phase I) and are used to monitor everything after it.")
        c1, c2, c3 = st.columns([1.1, 1.3, 2.4])
        chart_kind = c1.selectbox("Chart type", ["X̄-R (subgroups)", "I-MR (individuals)"])
        baseline = c2.slider("Baseline subgroups", 10, k, min(30, k), key="baseline_sub",
                             help="The first N subgroups are used to compute the control limits.")
        rules = c3.multiselect("Run rules", list(spc.RULE_LABELS), default=list(spc.RULE_LABELS),
                               format_func=lambda r: f"{r} · {spc.RULE_LABELS[r]}")

        is_xbar = chart_kind.startswith("X̄")
        if is_xbar:
            cc = spc.xbar_r_chart(matrix, baseline, rules)
            x_label, s_label, x_title, x_idx = "Subgroup mean (X̄)", "Range (R)", "Subgroup", list(range(1, k + 1))
        else:
            flat = matrix.ravel()
            cc = spc.imr_chart(flat, baseline * n, rules)
            x_label, s_label, x_title, x_idx = "Individual value", "Moving range", "Observation", list(range(1, len(flat) + 1))

        flagged = sorted(cc.signals)
        unit = "subgroup" if is_xbar else "observation"
        in_control = not flagged and not cc.spread_signals
        ui.show('<div style="margin:12px 0 4px">' + (
            ui.pill("IN CONTROL", "mint") + " " + ui.pill("no signals detected", "neutral", dot=False) if in_control else
            ui.pill("OUT OF CONTROL", "red") + " " + ui.pill(f"{len(flagged)} signal points · {len(cc.spread_signals)} range signals", "amber", dot=False)
        ) + "</div>")

        cols = st.columns(5)
        for col, (label, value, sub, tone) in zip(cols, [
            ("Center line", f"{cc.cl:.4f}", f"σ estimate {cc.sigma_within:.4f}", "mint"),
            ("Upper limit", f"{cc.ucl:.4f}", "UCL = CL + 3σ", "red"),
            ("Lower limit", f"{cc.lcl:.4f}", "LCL = CL − 3σ", "red"),
            ("Signal points", f"{len(flagged)}", f"of {len(x_idx)} {unit}s", "amber" if flagged else "mint"),
            ("Range signals", f"{len(cc.spread_signals)}", "variation chart", "amber" if cc.spread_signals else "mint")]):
            col.markdown(ui.kpi(label, value, sub, tone), unsafe_allow_html=True)

        st.write("")
        plot(charts.control_chart(cc, x_idx, x_label, x_title))
        plot(charts.spread_chart(cc, x_idx, s_label))

        # Insights
        by_rule: dict[str, int] = {}
        for i in flagged:
            for r in cc.signals[i]:
                by_rule[r] = by_rule.get(r, 0) + 1
        i1, i2, i3 = st.columns(3)
        if flagged:
            first = flagged[0]
            i1.markdown(ui.insight("First signal", f"At <b>{unit} {x_idx[first]}</b> "
                                   f"({', '.join(cc.signals[first])}). Investigate what changed around then.", "red"), unsafe_allow_html=True)
            parts = " · ".join(f"<b>{r}</b> ×{c}" for r, c in sorted(by_rule.items()))
            i2.markdown(ui.insight("Rules triggered", f"{parts}. A point can trigger several rules at once.", "amber"), unsafe_allow_html=True)
        else:
            i1.markdown(ui.insight("First signal", "None. Every point is inside the limits and no run rule fired.", "mint"), unsafe_allow_html=True)
            i2.markdown(ui.insight("Rules triggered", "None of the selected rules fired.", "mint"), unsafe_allow_html=True)
        if cc.spread_signals:
            pts = ", ".join(str(x_idx[i]) for i in cc.spread_signals[:6]) + ("…" if len(cc.spread_signals) > 6 else "")
            i3.markdown(ui.insight("Variation", f"The {s_label.lower()} chart exceeds its limit at <b>{pts}</b>: "
                                   "an outlier or a spell of unstable variation.", "violet"), unsafe_allow_html=True)
        else:
            i3.markdown(ui.insight("Variation", "Short-term variation is stable: no point exceeds the upper range limit.", "mint"), unsafe_allow_html=True)

        if flagged:
            with st.expander(f"View all {len(flagged)} signal points"):
                rows = [{unit.capitalize(): x_idx[i], "Value": round(float(cc.x[i]), 4), "Rules": ", ".join(cc.signals[i])}
                        for i in flagged]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=260)

# ═══════════════════════════════ CAPABILITY ═════════════════════════════════
with tab_cap:
    if spc_error:
        st.error(spc_error)
    else:
        flat = matrix.ravel()
        cc_cap = spc.xbar_r_chart(matrix, st.session_state.get("baseline_sub", min(30, matrix.shape[0])))
        if source.startswith("Sample"):
            d_lsl, d_usl = 24.960, 25.040
        else:
            half = 4 * float(flat.std(ddof=1))
            d_lsl, d_usl = float(round(flat.mean() - half, 3)), float(round(flat.mean() + half, 3))

        ui.section("Process capability", "Does the process meet specification?",
                   "Short-term indices (Cp, Cpk) use the within-subgroup sigma; long-term indices (Pp, Ppk) use the overall sigma.")
        a, b, _ = st.columns([1, 1, 2.4])
        lsl = a.number_input("LSL · lower spec limit", value=d_lsl, format="%.4f", step=0.001)
        usl = b.number_input("USL · upper spec limit", value=d_usl, format="%.4f", step=0.001)
        if source.startswith("Upload"):
            st.caption("Defaults for uploaded data are placeholders (mean ± 4σ). Enter your real specification limits.")

        if usl <= lsl:
            st.error("USL must be greater than LSL.")
        else:
            cap = spc.capability(flat, cc_cap.sigma_within, lsl, usl)
            tone = lambda v: "red" if v < 1 else "amber" if v < 1.33 else "mint"
            ui.show('<div style="margin:6px 0 2px">' + ui.pill(f"Short-term · {spc.verdict(cap.cpk)}", tone(cap.cpk)) + " "
                    + ui.pill(f"Long-term · {spc.verdict(cap.ppk)}", tone(cap.ppk)) + "</div>")
            plot(charts.capability_gauges(cap))

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.kpi("Process mean", f"{cap.mean:.4f}", f"spec centre {(lsl + usl) / 2:.4f}", "cyan"), unsafe_allow_html=True)
            k2.markdown(ui.kpi("σ within", f"{cap.sigma_within:.4f}", "from R̄ / d₂ (baseline)", "violet"), unsafe_allow_html=True)
            k3.markdown(ui.kpi("σ overall", f"{cap.sigma_overall:.4f}", "all measurements", "violet"), unsafe_allow_html=True)
            k4.markdown(ui.kpi("Expected out of spec", f"{cap.ppm_overall:,.0f}", "parts per million (long-term)",
                               "red" if cap.ppm_overall > 1000 else "amber" if cap.ppm_overall > 63 else "mint", unit="ppm"), unsafe_allow_html=True)

            st.write("")
            if cap.cp - cap.cpk > 0.15:
                ui.callout(f"<b>Off-center.</b> Cp − Cpk = {cap.cp - cap.cpk:.2f}. Re-centering the mean would recover capability.", "amber", "🎯")
            if cap.cpk - cap.ppk > 0.15:
                ui.callout(f"<b>Drift between subgroups.</b> Cpk is {cap.cpk:.2f} but Ppk is {cap.ppk:.2f}: shifts or drift are inflating the "
                           "overall variation. Fix stability first, then re-check capability.", "red", "📉")
            if cc_cap.signals or cc_cap.spread_signals:
                ui.callout(f"<b>Process not in statistical control</b> ({len(cc_cap.signals)} signal points). "
                           "Capability indices assume a stable process, so read them as indicative only.", "cyan", "ℹ️")
            st.write("")
            plot(charts.capability_histogram(flat, cap, lsl, usl))

# ═══════════════════════════════ OEE ════════════════════════════════════════
with tab_oee:
    try:
        table = oee.shift_table(prod_df, down_df)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        table = None
    if table is not None:
        ui.section("Equipment effectiveness", "Where does productive time go?",
                   "OEE = Availability × Performance × Quality, aggregated time-weighted across the selected shifts.")
        shifts = oee.sort_shifts(table["shift"].unique().tolist())
        sel = st.multiselect("Shifts", shifts, default=shifts)
        t = table[table["shift"].isin(sel)]
        d = down_df[down_df["shift"].isin(sel)]
        if t.empty:
            st.warning("Select at least one shift.")
        else:
            r = oee.ratios(t)
            g1, g2 = st.columns([1.05, 1.25])
            with g1:
                plot(charts.oee_gauge(r["oee"], oee.BENCHMARK))
            with g2:
                gap = (r["oee"] - oee.BENCHMARK) * 100
                st.markdown(ui.bar_card("Availability", r["availability"], "cyan", "Run time ÷ planned time"), unsafe_allow_html=True)
                st.write("")
                st.markdown(ui.bar_card("Performance", r["performance"], "violet", "Actual output ÷ ideal output while running"), unsafe_allow_html=True)
                st.write("")
                st.markdown(ui.bar_card("Quality", r["quality"], "green", "Good parts ÷ total parts"), unsafe_allow_html=True)
                st.write("")
                ui.callout(f"OEE is <b>{abs(gap):.1f} points {'below' if gap < 0 else 'above'}</b> the commonly quoted 85% benchmark.",
                           "amber" if gap < 0 else "mint", "🏁")

            w = oee.loss_waterfall(t)
            plot(charts.waterfall(w))

            p1, p2 = st.columns(2)
            pareto_df = oee.pareto(d)
            with p1:
                plot(charts.pareto(pareto_df))
            with p2:
                plot(charts.daily_trend(oee.daily_oee(t), oee.BENCHMARK))

            bs = oee.by_shift(t)
            plot(charts.shift_bars(bs))

            losses = {"downtime (availability)": w["downtime"], "slow running (performance)": w["speed"], "scrap (quality)": w["quality"]}
            biggest = max(losses, key=losses.get)
            share = losses[biggest] / sum(losses.values()) * 100
            n80 = int((pareto_df["cum_pct"] < 80).sum()) + 1 if len(pareto_df) else 0
            i1, i2, i3 = st.columns(3)
            i1.markdown(ui.insight("Biggest loss", f"<b>{biggest.capitalize()}</b> takes {share:.0f}% of all lost time "
                                   f"({losses[biggest] / 60:,.0f} h).", "red"), unsafe_allow_html=True)
            if n80:
                names = ", ".join(pareto_df["reason"].head(n80))
                i2.markdown(ui.insight("Vital few", f"<b>{names}</b> explain {pareto_df['cum_pct'].iloc[n80 - 1]:.0f}% of downtime. "
                                       "Start improvement here.", "amber"), unsafe_allow_html=True)
            if len(bs) > 1:
                ranked = bs.sort_values("oee")
                worst, best = ranked.iloc[0], ranked.iloc[-1]
                i3.markdown(ui.insight("Shift gap", f"<b>{worst['shift']}</b> runs at {worst['oee'] * 100:.0f}% OEE versus "
                                       f"<b>{best['shift']}</b> at {best['oee'] * 100:.0f}%: a {(best['oee'] - worst['oee']) * 100:.0f}-point gap.", "violet"),
                            unsafe_allow_html=True)

# ═══════════════════════════════ METHOD ═════════════════════════════════════
with tab_method:
    ui.section("Method", "How the numbers are computed", "Every formula is implemented from scratch and covered by unit tests.")
    m1, m2 = st.columns(2)
    with m1:
        with st.container(border=True):
            st.markdown("##### Control charts")
            st.latex(r"UCL,\,LCL = \bar{\bar{X}} \pm A_2 \bar{R} \qquad \sigma = \bar{R} / d_2")
            st.latex(r"UCL_R = D_4 \bar{R} \qquad LCL_R = D_3 \bar{R}")
            st.markdown("**I-MR:** limits = X̄ ± 3 · (MR̄ / 1.128). Limits are computed from a baseline window (Phase I) "
                        "and then applied to the remaining data (Phase II).")
            st.markdown("**Run rules (Western Electric / Nelson)**  \n"
                        "`R1` one point beyond 3σ · `R2` nine in a row on one side · `R3` six in a row rising or falling · "
                        "`R5` two of three beyond 2σ on the same side.")
    with m2:
        with st.container(border=True):
            st.markdown("##### Process capability")
            st.latex(r"C_p = \frac{USL - LSL}{6\sigma_{within}} \qquad C_{pk} = \frac{\min(USL-\mu,\ \mu-LSL)}{3\sigma_{within}}")
            st.latex(r"P_p,\ P_{pk}:\ \text{same formulas with } \sigma_{overall}")
            st.markdown("A large gap between **Cpk** and **Ppk** means shifts or drift *between* subgroups. "
                        "Indices are only trustworthy when the process is stable. "
                        "Reading: Cpk ≥ 1.33 capable · 1.00 to 1.33 marginal · below 1.00 not capable.")
    with st.container(border=True):
        st.markdown("##### OEE")
        st.latex(r"OEE = A \times P \times Q \qquad A=\frac{\text{run time}}{\text{planned time}},\ \ "
                 r"P=\frac{\text{ideal cycle time} \times \text{total count}}{\text{run time}},\ \ Q=\frac{\text{good count}}{\text{total count}}")
        st.markdown("Aggregates are **time-weighted**: sums first, then ratios. The 85% benchmark is a commonly quoted reference, "
                    "not a universal target.")

ui.footer()
