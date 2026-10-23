import pandas as pd

import charts
import oee
import spc


def _data():
    ids, m = spc.to_subgroup_matrix(pd.read_csv("data/sample_spc.csv"))
    cc = spc.xbar_r_chart(m, 30)
    cap = spc.capability(m.ravel(), cc.sigma_within, 24.96, 25.04)
    prod, down = pd.read_csv("data/sample_production_log.csv"), pd.read_csv("data/sample_downtime_log.csv")
    return m, cc, cap, oee.shift_table(prod, down), down


def test_all_charts_build():
    m, cc, cap, t, down = _data()
    x_idx = list(range(1, len(cc.x) + 1))
    figs = [charts.control_chart(cc, x_idx, "X", "Subgroup"), charts.spread_chart(cc, x_idx, "R"),
            charts.capability_gauges(cap), charts.capability_histogram(m.ravel(), cap, 24.96, 25.04),
            charts.oee_gauge(oee.ratios(t)["oee"]), charts.waterfall(oee.loss_waterfall(t)),
            charts.pareto(oee.pareto(down)), charts.daily_trend(oee.daily_oee(t)),
            charts.shift_bars(oee.by_shift(t))]
    assert all(len(f.data) > 0 for f in figs)


def test_imr_chart_builds():
    m, *_ = _data()
    cc = spc.imr_chart(m.ravel(), 150)
    fig = charts.control_chart(cc, list(range(1, len(cc.x) + 1)), "Individual", "Observation")
    assert len(fig.data) >= 1


def test_sample_data_tells_the_expected_story():
    m, cc, cap, t, down = _data()
    assert len(cc.signals) > 20                      # shift + drift are detected
    assert cap.cpk - cap.ppk > 0.15                  # short-term looks better than long-term
    assert 0.65 < oee.ratios(t)["oee"] < 0.75


def test_shift_order():
    assert oee.sort_shifts(["Night", "Morning", "Afternoon"]) == ["Morning", "Afternoon", "Night"]
    assert oee.sort_shifts(["B", "Night", "A"]) == ["Night", "A", "B"]
