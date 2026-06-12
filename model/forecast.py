#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBP/CNY 换汇时机预测引擎  (Phase 1+2)
- 数据: frankfurter.dev 5年日频 GBP/CNY, GBP/USD, USD/CNY
- 模型: 随机游走(基准) + ARIMA + ETS + LightGBM(含外生变量)  -> 集成漂移
        GARCH(1,1) -> 条件波动率
- 蒙特卡洛: 5000 条未来30天路径 -> P(触及目标价)/预期最低/扇形区间
- 回测: walk-forward, 跟随机游走比, 区间校准
输出: ../predictions.json  和  ../backtest_report.md
诚实立场: 方向几乎不可预测; 价值在量化概率与区间, 模型打不过基准就如实标注.
"""
import os, sys, json, math, datetime, warnings, urllib.request
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TARGET = float(os.environ.get("FX_TARGET", "9.03"))
HORIZONS = [1, 3, 7, 14, 30]
N_SIM = 5000
N_DAYS = 30
SEED = 42

# ---------------- 数据 ----------------
def fetch_series(frm, to, start, end):
    url = f"https://api.frankfurter.dev/v1/{start}..{end}?from={frm}&to={to}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = json.loads(urllib.request.urlopen(req, timeout=40).read())
    rates = d["rates"]
    s = pd.Series({pd.Timestamp(k): v[to] for k, v in rates.items()}).sort_index()
    return s

def load_data():
    end = datetime.date.today()
    start = end.replace(year=end.year - 5)
    a, b = start.isoformat(), end.isoformat()
    df = pd.DataFrame({
        "gbpcny": fetch_series("GBP", "CNY", a, b),
        "gbpusd": fetch_series("GBP", "USD", a, b),
        "usdcny": fetch_series("USD", "CNY", a, b),
    }).dropna()
    return df

# ---------------- 特征 ----------------
def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def build_features(df):
    x = pd.DataFrame(index=df.index)
    p = df["gbpcny"]
    r = np.log(p).diff()
    x["r1"] = r
    x["r2"] = r.shift(1)
    x["r3"] = r.shift(2)
    x["r5"] = r.shift(4)
    x["r10"] = r.shift(9)
    x["ma5"] = (p / p.rolling(5).mean() - 1)
    x["ma20"] = (p / p.rolling(20).mean() - 1)
    x["vol20"] = r.rolling(20).std()
    x["rsi"] = rsi(p) / 100
    x["dow"] = df.index.dayofweek
    rng_hi = p.rolling(20).max(); rng_lo = p.rolling(20).min()
    x["pos20"] = (p - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)
    x["gbpusd_r"] = np.log(df["gbpusd"]).diff()      # 外生: 英镑兑美元日收益
    x["usdcny_r"] = np.log(df["usdcny"]).diff()      # 外生: 美元兑人民币日收益
    y_next = r.shift(-1)                               # 预测下一日 log 收益
    return x, y_next, r

# ---------------- 各模型: 预测下一日 log 收益 ----------------
def pred_rw(r, win=60):
    return float(r.dropna().tail(win).mean())

def pred_arima(logp):
    from statsmodels.tsa.arima.model import ARIMA
    try:
        m = ARIMA(logp.values, order=(1, 1, 1)).fit()
        fc = m.forecast(1)[0]
        return float(fc - logp.values[-1])
    except Exception:
        return np.nan

def pred_ets(logp):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        m = ExponentialSmoothing(logp.values, trend="add", damped_trend=True).fit()
        fc = m.forecast(1)[0]
        return float(fc - logp.values[-1])
    except Exception:
        return np.nan

def fit_lgbm(X, y):
    import lightgbm as lgb
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    if len(d) < 200:
        return None, None
    feats = list(X.columns)
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.02, num_leaves=15,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, verbose=-1)
    model.fit(d[feats], d["y"])
    return model, feats

def pred_lgbm(model, feats, X):
    if model is None:
        return np.nan
    row = X.iloc[[-1]][feats]
    if row.isnull().any(axis=1).iloc[0]:
        row = row.fillna(0)
    return float(model.predict(row)[0])

# ---------------- GARCH 波动率 ----------------
def garch_vol_path(r, n_days):
    """返回 (未来n天每日sigma, 标准化残差样本)  单位: log收益."""
    from arch import arch_model
    rr = r.dropna() * 100.0  # 百分比, 数值稳定
    am = arch_model(rr, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=n_days, reindex=False)
    var = fc.variance.values[-1]           # 百分比^2
    sigma = np.sqrt(var) / 100.0           # 回到 log 收益单位
    std_resid = res.std_resid.dropna().values
    return sigma, std_resid

# ---------------- 蒙特卡洛 ----------------
def monte_carlo(spot, mu, sigma_path, std_resid, target):
    rng = np.random.default_rng(SEED)
    z = rng.choice(std_resid, size=(N_SIM, N_DAYS), replace=True)
    daily = mu + sigma_path[np.newaxis, :] * z      # (N_SIM, N_DAYS) log收益
    logpaths = np.log(spot) + np.cumsum(daily, axis=1)
    paths = np.exp(logpaths)                          # 价格路径
    qs = [10, 25, 50, 75, 90]
    fan = []
    today = pd.Timestamp.today().normalize()
    for d in range(N_DAYS):
        col = paths[:, d]
        date = (today + pd.Timedelta(days=d + 1)).date().isoformat()
        fan.append({"date": date, **{f"p{q}": round(float(np.percentile(col, q)), 4) for q in qs}})
    out = {"fan": fan}
    prob_hit, exp_min, hz = {}, {}, {}
    for H in HORIZONS:
        sub = paths[:, :H]
        mins = sub.min(axis=1)
        prob = float((mins <= target).mean())
        prob_hit[str(H)] = round(prob, 3)
        exp_min[str(H)] = {
            "mean": round(float(mins.mean()), 4),
            "p25": round(float(np.percentile(mins, 25)), 4),
            "p50": round(float(np.percentile(mins, 50)), 4),
        }
        endcol = paths[:, H - 1]
        hz[str(H)] = {
            "p10": round(float(np.percentile(endcol, 10)), 4),
            "p50": round(float(np.percentile(endcol, 50)), 4),
            "p90": round(float(np.percentile(endcol, 90)), 4),
            "prob_hit": round(prob, 3),
        }
    out.update(prob_hit=prob_hit, exp_min=exp_min, horizons=hz)
    return out

# ---------------- 回测 (walk-forward) ----------------
def backtest(df, X, y_next, r, n_test=252, refit_every=21):
    logp = np.log(df["gbpcny"])
    idx = df.index
    n = len(df)
    start = n - n_test
    rows = []
    lgbm_model = lgbm_feats = None
    arima_pred = ets_pred = np.nan
    for i in range(start, n - 1):
        actual = float(y_next.iloc[i])  # 真实下一日收益
        if np.isnan(actual):
            continue
        hist_logp = logp.iloc[: i + 1]
        # 基准
        rw = float(r.iloc[max(0, i - 59): i + 1].mean())
        # 重型模型按周期重拟合
        if (i - start) % refit_every == 0:
            arima_pred = pred_arima(hist_logp)
            ets_pred = pred_ets(hist_logp)
            lgbm_model, lgbm_feats = fit_lgbm(X.iloc[: i + 1], y_next.iloc[: i + 1])
        lg = pred_lgbm(lgbm_model, lgbm_feats, X.iloc[: i + 1]) if lgbm_model else np.nan
        sig20 = float(r.iloc[max(0, i - 19): i + 1].std())
        rows.append((actual, rw, arima_pred, ets_pred, lg, sig20))
    bt = pd.DataFrame(rows, columns=["actual", "rw", "arima", "ets", "lgbm", "sig20"])
    def metrics(pred_col):
        m = bt[["actual", pred_col, "sig20"]].dropna()
        if len(m) < 20:
            return None
        err = m[pred_col] - m["actual"]
        rmse = float(np.sqrt((err ** 2).mean()))
        dir_acc = float((np.sign(m[pred_col]) == np.sign(m["actual"])).mean())
        return {"rmse": round(rmse, 6), "dir_acc": round(dir_acc, 3), "n": int(len(m))}
    res = {k: metrics(k) for k in ["rw", "arima", "ets", "lgbm"]}
    # 区间校准: 80% 区间 = pred_drift ± 1.2816*sig20, 用最优均值模型(rw)
    m = bt.dropna(subset=["actual", "sig20"])
    lo = m["rw"] - 1.2816 * m["sig20"]; hi = m["rw"] + 1.2816 * m["sig20"]
    cov80 = float(((m["actual"] >= lo) & (m["actual"] <= hi)).mean())
    # 谁打过基准 (RMSE 更低)
    base = res["rw"]["rmse"] if res["rw"] else None
    beats = {k: (v is not None and base is not None and v["rmse"] < base) for k, v in res.items() if k != "rw"}
    return {"models": res, "coverage80": round(cov80, 3), "target_coverage": 0.80,
            "beats_baseline": beats, "n_test": int(len(bt))}

# ---------------- 主流程 ----------------
def main():
    print("拉取数据 ...")
    df = load_data()
    X, y_next, r = build_features(df)
    spot = float(df["gbpcny"].iloc[-1])
    data_date = df.index[-1].date().isoformat()
    logp = np.log(df["gbpcny"])
    print(f"spot={spot:.4f}  data_date={data_date}  rows={len(df)}")

    print("回测 walk-forward ...")
    bt = backtest(df, X, y_next, r)

    print("拟合当前各模型 (下一日漂移) ...")
    drift_models = {
        "rw": pred_rw(r),
        "arima": pred_arima(logp),
        "ets": pred_ets(logp),
    }
    lgbm_model, lgbm_feats = fit_lgbm(X, y_next)
    drift_models["lgbm"] = pred_lgbm(lgbm_model, lgbm_feats, X)

    # 集成漂移: 用回测 RMSE 倒数加权, 仅纳入"打过基准 或 就是基准"的模型; 强烈向 0 收缩
    weights, mu_num, mu_den = {}, 0.0, 0.0
    for k, v in drift_models.items():
        if v is None or np.isnan(v):
            continue
        rm = bt["models"].get(k)
        if rm is None:
            continue
        eligible = (k == "rw") or bt["beats_baseline"].get(k, False)
        if not eligible:
            continue
        w = 1.0 / (rm["rmse"] + 1e-9)
        weights[k] = w; mu_num += w * v; mu_den += w
    mu_raw = (mu_num / mu_den) if mu_den > 0 else drift_models["rw"]
    # 收缩 + 限幅: 漂移不可过度自信, 限制在 ±0.3 倍日波动内
    sig_now = float(r.dropna().tail(20).std())
    mu = float(np.clip(0.5 * mu_raw, -0.3 * sig_now, 0.3 * sig_now))

    print("GARCH 波动率 + 蒙特卡洛 ...")
    sigma_path, std_resid = garch_vol_path(r, N_DAYS)
    mc = monte_carlo(spot, mu, sigma_path, std_resid, TARGET)

    # 建议文案
    p14 = mc["prob_hit"]["14"]; p30 = mc["prob_hit"]["30"]
    emin14 = mc["exp_min"]["14"]["mean"]
    any_beats = any(bt["beats_baseline"].values())
    if spot <= TARGET:
        rec = f"现价 {spot:.4f} 已≤目标 {TARGET}，建议直接出手。"
    elif p14 >= 0.5:
        rec = f"14天内触及 {TARGET} 概率 {p14:.0%}（预期最低≈{emin14}），值得挂单等。"
    elif p30 >= 0.5:
        rec = f"14天内触及概率仅 {p14:.0%}，但30天 {p30:.0%}；可耐心等或分批换。"
    else:
        rec = f"30天内触及 {TARGET} 概率仅 {p30:.0%}，目标偏低；要么降低目标，要么分批换以摊平。"
    honesty = ("方向回测：各模型未显著超越随机游走，方向视为不可预测；"
               if not any_beats else
               "方向回测：部分模型样本外略超基准，但提升有限；")
    honesty += f"80%区间校准实测覆盖 {bt['coverage80']:.0%}（目标80%），区间可" + ("信。" if abs(bt['coverage80']-0.8) <= 0.08 else "参考。")

    hist_tail = [{"d": d.date().isoformat(), "v": round(float(v), 4)}
                 for d, v in df["gbpcny"].tail(30).items()]

    out = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "data_date": data_date,
        "spot": round(spot, 4),
        "target": TARGET,
        "drivers": {"gbpusd": round(float(df["gbpusd"].iloc[-1]), 4),
                     "usdcny": round(float(df["usdcny"].iloc[-1]), 4)},
        "mu_daily": round(mu, 6),
        "ensemble_weights": {k: round(w / sum(weights.values()), 3) for k, w in weights.items()} if weights else {},
        "drift_models": {k: (round(v, 6) if v is not None and not np.isnan(v) else None)
                          for k, v in drift_models.items()},
        "history_tail": hist_tail,
        "fan": mc["fan"],
        "prob_hit": mc["prob_hit"],
        "exp_min": mc["exp_min"],
        "horizons": mc["horizons"],
        "backtest": bt,
        "recommendation": rec,
        "honesty": honesty,
        "disclaimer": "汇率方向短期不可预测；本系统量化概率与区间，不构成投资建议。",
    }
    with open(os.path.join(ROOT, "predictions.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_report(out)
    print(f"OK -> predictions.json  | P(14d≤{TARGET})={p14:.0%} P(30d)={p30:.0%} cov80={bt['coverage80']:.0%}")
    return 0

def write_report(o):
    bt = o["backtest"]; m = bt["models"]
    lines = [
        f"# GBP/CNY 预测回测报告", "",
        f"- 生成: {o['generated_at']}  数据日期: {o['data_date']}",
        f"- 现价: {o['spot']}  目标: {o['target']}  (1英镑={o['spot']}元)",
        f"- 驱动: GBP/USD={o['drivers']['gbpusd']}  USD/CNY={o['drivers']['usdcny']}",
        "", "## 触及目标价概率 (蒙特卡洛 5000 路径)", "",
        "| 天数 | P(≤目标) | 预期最低 |", "|---|---|---|",
    ]
    for H in ["7", "14", "30"]:
        lines.append(f"| {H} | {o['prob_hit'][H]:.0%} | {o['exp_min'][H]['mean']} |")
    lines += ["", "## 方向回测 (walk-forward, vs 随机游走)", "",
              f"测试样本: {bt['n_test']} 个交易日", "",
              "| 模型 | RMSE(收益) | 方向准确率 | 超基准? |", "|---|---|---|---|"]
    for k in ["rw", "arima", "ets", "lgbm"]:
        v = m.get(k)
        if not v:
            continue
        beats = "—(基准)" if k == "rw" else ("✅" if bt["beats_baseline"].get(k) else "❌")
        lines.append(f"| {k} | {v['rmse']} | {v['dir_acc']:.0%} | {beats} |")
    lines += ["", "- **区间校准**: 80% 区间实测覆盖 **{:.0%}** (目标 80%)".format(bt["coverage80"]),
              "", "## 结论", "", f"> {o['recommendation']}", "", f"> {o['honesty']}", "",
              f"_{o['disclaimer']}_", ""]
    with open(os.path.join(ROOT, "backtest_report.md"), "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    sys.exit(main())
