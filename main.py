"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (my own design, standard library only, deterministic, no network):

Exophase prints, for each trophy, the share of the players it tracks who have
earned it. That share is a ratio of two counts that only move when Exophase
refreshes the page: the players tracked, N, and the earners of each trophy,
E_i. So instead of extrapolating the printed percentages directly, this
program models the two counts separately and divides at the target:

    rate_i(target) = 100 * E_i(target) / N(target)

Both counts grow roughly as power laws in the time since release, so each is
fitted in log space against u = ln(1 + hours since release midnight), with
an exponentially recency-weighted least-squares line (a read's weight halves
every ~0.12 units of u, so at the deadline the fit is governed by roughly
the last day of the case's history). The strong recency weighting matters
for two reasons. First, earners saturate: a trophy everyone gets early
stops growing long before the target, and a fit over the whole history
would carry its steep early climb forward and forecast it near 100%.
Second, the tracked population itself changes character over the first
week -- the earliest trackers are the most committed players, and their
earning slopes describe a mix that no longer holds by the deadline; the
slopes that persist to the target are the recent ones. (The half-life was
stress-tested against simulated week-one evolutions of the page -- cohort
models with drifting player mix, saturating maturity curves, stale-page
refresh timing and observation noise: shortening it from 0.55 to ~0.12
lowered the mean pinball loss on every simulated world, and the bands
below were widened slightly to offset the narrower local fits' smaller
residuals.) Forecasts are anchored on the last read, not on the fitted
intercept, so a noisy early history cannot drag the level away from what
the page actually says now.
Per-trophy earner slopes are shrunk toward the cross-trophy median slope,
because a trophy's own slope is noisy when few players hold it, and slopes
are clamped so earners never shrink and never outrun players by much.

When the history is flat -- every read identical, as on launch night before
Exophase's first refresh -- no slope can be estimated at all. Then the
program falls back to a refresh mixture: with some probability the page is
still unrefreshed at the target (rate unchanged), and otherwise one refresh
lands before the target and dilutes every rate. The dilution is NOT a
uniform discount: the launch-night refresh the rules describe proves the
new players a refresh adds hold almost nothing beyond the one trophy
everyone holds. Tracked players went 638 -> 3,868 (+3,230) and trophies
earned 943 -> 4,276 (+3,333); the trophy at 100% stayed at 100%, so it
alone accounts for +3,230 of the +3,333, leaving only +103 trophies spread
over the other 54 -- about 0.03 trophies per new player beyond the first
one. So a refresh is modelled as: every new player holds the saturated
trophies, and holds each other trophy at a rate proportional to its current
rate, scaled so the non-saturated trophies total ~3.2 per 100 new players.
The refresh probability grows with the horizon and with how stale the page
already is, against a ~4.5 hour refresh interval (the first refresh came
after ~4.5 hours of staleness).

Spreads are honest logit-normal bands: a base term for refresh timing and
model error, a term that grows with the square root of the horizon, the
residual scatter of each trophy's own fit, and a wider band for trophies
held by very few players. Quantiles are the median shifted by the normal
z-scores at the scored levels.

Input/output follow the competition: read every <id>.in in $MAKEFUN_INPUT_DIR
(or /input), write <id>.out with {"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]} to $MAKEFUN_OUTPUT_DIR (or /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Priors calibrated on the launch-night refresh described in the rules ---
# Non-saturated trophies held per 100 newly tracked players at a refresh:
# (4276 - 943 - 3230) / 3230 * 100 = 3.19 -- the trophy at 100% accounts
# for 3,230 of the 3,333 trophies the refresh added, so the other 54 share
# only ~103. A uniform per-trophy discount is refuted by these aggregates:
# it would predict +4,310 trophies, not +3,333.
FRESH_NON_INTRO = 3.2
# Tracked players grew ~1.05 log units per unit of ln(1 + hours since release):
# ln(3868/638) / ln(1 + 4.5) = 1.802 / 1.705.
B_PRIOR = 1.05
# Prior median gap between Exophase refreshes, in hours (the first refresh
# came after ~4.5 hours of the page sitting unchanged).
R_REFRESH_H = 4.5

# --- Spread parameters, in logit units ---
# Widened ~15% from the original 0.30 / 0.35: the strongly recency-weighted
# fits track recent dynamics better but their residuals understate how far
# a 4.5-day extrapolation can miss, so the base bands compensate.
S_BASE = 0.345       # refresh timing and model error, always present
S_HORIZON = 0.4025   # per sqrt(day) of horizon
S_MAX = 1.40
S_MIN = 0.30

# A trophy held by everyone gets a tight band near 100%.
S_SATURATED = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, so outputs are short and stable."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (fractional kept)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def release_midnight(case):
    """Epoch seconds of 00:00 UTC on the game's release date."""
    day = case.get("game", {}).get("release_date", "")
    try:
        dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def u_of(epoch, midnight):
    """ln(1 + hours since release midnight); 0 before/at midnight."""
    if midnight is None:
        return 0.0
    hours = (epoch - midnight) / 3600.0
    return math.log1p(hours) if hours > 0 else 0.0


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, residual std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        resid = math.sqrt(
            sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
            / (sw * (n - 2) / n)
        )
    else:
        resid = 0.0
    return (a, b, resid)


def distinct_snapshots(case, midnight):
    """Collapse history to reads where players or any rate actually changed.

    Exophase only moves the page at its own refreshes, so long flat stretches
    are one observation, not many.
    """
    snaps = []
    prev_key = None
    for read in case.get("history", []):
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(
            {
                "u": u_of(parse_time(read["at"]), midnight),
                "at": parse_time(read["at"]),
                "n": read.get("players_tracked") or 0,
                "rates": list(read.get("rates") or ()),
            }
        )
    return snaps


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(centers, sigmas, weights, level):
    """Quantile of a mixture of logit-normals, by bisection on its CDF."""
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = sum(
            w * phi((mid - c) / s)
            for c, s, w in zip(centers, sigmas, weights)
        )
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_flat(case, snaps, midnight):
    """History never moved: anticipate the next refresh, or none at all.

    Two modes: the page stays as it is (probability 1 - p), or one refresh
    lands before the target (probability p) and dilutes every rate. A refresh
    catches up all the player growth since the page went stale, so its size
    is computed from the growth since the first read, not just since the
    checkpoint. New players hold every saturated trophy, and each other
    trophy at a rate proportional to its current rate, scaled so the
    non-saturated trophies total FRESH_NON_INTRO per 100 new players.
    """
    last = snaps[-1]
    horizon_h = case["horizon_minutes"] / 60.0
    # How long the page has sat unchanged: from the first read to the
    # checkpoint. (Distinct snapshots collapse the flat stretch to one, so
    # this must use the case's checkpoint, not the last snapshot.)
    checkpoint = parse_time(case["at"]) if case.get("at") else last["at"]
    stale_h = max(0.0, (checkpoint - snaps[0]["at"]) / 3600.0)
    p_refresh = 1.0 - math.exp(-(horizon_h + 0.5 * stale_h) / R_REFRESH_H)
    p_refresh = min(0.9, max(0.05, p_refresh))

    u_first = snaps[0]["u"]
    u_target = u_of(parse_time(case["target"]), midnight)
    growth = math.exp(B_PRIOR * min(u_target - u_first, 2.5))
    n_last = max(1, last["n"])
    n_new = n_last * (growth - 1.0)

    # Last non-null rate per trophy (the last read may lack one).
    last_rates = []
    for i in range(len(case["trophies"])):
        rate = None
        for s in reversed(snaps):
            if i < len(s["rates"]) and s["rates"][i] is not None:
                rate = s["rates"][i]
                break
        last_rates.append(rate if rate is not None else 1.0)

    non_sat_total = sum(r for r in last_rates if r < 99.99)
    k = FRESH_NON_INTRO / non_sat_total if non_sat_total > 0 else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        if rate >= 99.99:
            # Everyone tracked holds it; a refresh cannot dilute it.
            lg = logit(rate)
            qs = [round4(clamp_pct(inv_logit(lg + z * S_SATURATED))) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        # Earners after a refresh: current earners plus the new players'
        # holdings of this trophy (k * rate percent of the new players).
        e_after = rate * n_last / 100.0 + (k * rate / 100.0) * n_new
        rate_after = 100.0 * e_after / (n_last + n_new)
        mu_same = logit(rate)
        mu_ref = logit(rate_after)
        qs = []
        for level in LEVELS:
            lg = mixture_quantile(
                [mu_same, mu_ref], [0.05, 0.30], [1.0 - p_refresh, p_refresh], level
            )
            qs.append(round4(clamp_pct(inv_logit(lg))))
        # Rounding can only move a value by <0.0001; enforce non-decreasing.
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_rich(case, snaps, midnight):
    """History moved: model earners and players as power laws, divide."""
    n_trophies = len(case["trophies"])
    last = snaps[-1]
    u_last = last["u"]
    u_target = u_of(parse_time(case["target"]), midnight)
    du = max(0.0, u_target - u_last)
    days = case["horizon_minutes"] / 1440.0

    us = [s["u"] for s in snaps]
    # Recency weights: a read's weight halves every ~0.12 units of u
    # (roughly the last day of the case's history at the deadline). Trophy
    # earners saturate -- a trophy everyone gets early stops growing long
    # before the target -- so the slope that matters is the recent one:
    # fitting the whole history would carry the steep early climb forward
    # and forecast saturated trophies near 100%. The recent window also
    # reflects the current tracked-player mix; launch-phase slopes come
    # from a far more hardcore population and extrapolate badly.
    W_HALF = 0.12
    ws = [math.exp(-(u_last - u) * math.log(2) / W_HALF) for u in us]

    # Players tracked: log N ~ a + b*u.
    log_n = [math.log(max(1, s["n"])) for s in snaps]
    _, b_n, _ = wols(us, log_n, ws)
    b_n = min(2.0, max(0.0, b_n))
    n_target = max(1, last["n"]) * math.exp(b_n * du)

    # Per-trophy earner slopes, then shrink toward the cross-trophy median.
    # The shrinkage weight uses the fit's effective number of points, so a
    # trophy whose history barely moved in the recent window borrows the
    # median slope instead of trusting its own noise.
    slopes = []
    per_trophy = []
    for i in range(n_trophies):
        pts = [
            (s["u"], (s["rates"][i] * s["n"] / 100.0) if i < len(s["rates"]) and s["rates"][i] is not None else None)
            for s in snaps
        ]
        pts = [(u, e) for (u, e) in pts if e is not None]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [math.log(p[1] + 0.5) for p in pts]
            wts = [math.exp(-(u_last - x) * math.log(2) / W_HALF) for x in xs]
            _, b_i, resid = wols(xs, ys, wts)
            eff = sum(wts) ** 2 / sum(w * w for w in wts)
            slopes.append(b_i)
            per_trophy.append((pts[-1][1], b_i, resid, eff))
        elif len(pts) == 1:
            per_trophy.append((pts[0][1], None, 0.0, 0.0))
        else:
            per_trophy.append((0.0, None, 0.0, 0.0))
    b_med = median(slopes) if slopes else b_n
    b_med = min(2.0, max(0.0, b_med))

    # Saturating-rate fits, per trophy. Two two-parameter families, each
    # fitted by a grid search over tau with the cap solved by weighted
    # least squares:
    #   exponential: r(t) = cap * (1 - exp(-t / tau))  (fast approach)
    #   Hill:        r(t) = cap * t / (t + tau)        (slow, 1/t approach)
    # A trophy's earners saturate -- once most players who will ever reach
    # it have done so, its rate flattens -- and a power law extrapolated
    # from its early climb would forecast it near 100%. The saturating
    # fits cap that: the lower of the trusted fits is used, and it only
    # ever lowers the forecast, never raises it. The Hill form matters
    # near 100%: a trophy at 96% creeps toward 100% slowly, and the
    # exponential form would call it 100% at the target, which the logit
    # scale punishes severely if the truth is 98%.
    t_hours = [
        max(0.0, (s["at"] - midnight) / 3600.0) if midnight is not None else 0.0
        for s in snaps
    ]
    t_target = (
        max(0.0, (parse_time(case["target"]) - midnight) / 3600.0)
        if midnight is not None
        else 0.0
    )
    tau_grid = [0.5 * (10000.0 ** (k / 47.0)) for k in range(48)]

    def fit_family(pts, shape):
        """Best (residual, tau, cap) for r(t) = cap * shape(t, tau)."""
        best = None
        for tau in tau_grid:
            xs = [shape(t, tau) for (t, _, _) in pts]
            num = sum(w * x * r for x, (_, r, w) in zip(xs, pts))
            den = sum(w * x * x for x, (_, _, w) in zip(xs, pts))
            if den <= 0:
                continue
            cap = num / den
            if cap <= 0 or cap > 100.0:
                continue
            resid = 0.0
            wsum = 0.0
            for x, (t, r, w) in zip(xs, pts):
                pred = cap * x
                if pred <= 0:
                    continue
                resid += w * (math.log(max(r, 0.005)) - math.log(pred)) ** 2
                wsum += w
            if wsum <= 0:
                continue
            resid = math.sqrt(resid / wsum)
            if best is None or resid < best[0]:
                best = (resid, tau, cap)
        return best

    def saturating_forecast(i):
        pts = [
            (t_hours[j], snaps[j]["rates"][i], ws[j])
            for j in range(len(snaps))
            if i < len(snaps[j]["rates"]) and snaps[j]["rates"][i] is not None
        ]
        pts = [(t, r, w) for (t, r, w) in pts if r > 0.0]
        if len(pts) < 4:
            return None
        rates_only = [r for (_, r, _) in pts]
        if rates_only[-1] < 0.1:
            return None
        # Only for a rate that is not falling over the recent reads.
        if sum(rates_only[-2:]) / 2.0 < sum(rates_only[-4:-2]) / 2.0:
            return None
        cands = []
        for shape in (
            lambda t, tau: 1.0 - math.exp(-t / tau),
            lambda t, tau: t / (t + tau) if t > 0 else 0.0,
        ):
            best = fit_family(pts, shape)
            if best is None or best[0] > 0.4:
                continue
            resid, tau, cap = best
            if cap < rates_only[-1]:
                continue
            cands.append(cap * shape(t_target, tau))
        if not cands:
            return None
        return min(cands)

    sat = [saturating_forecast(i) for i in range(n_trophies)]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        e_last, b_i, resid, eff_n = per_trophy[i]
        if b_i is None:
            b_star = b_med
        else:
            w = eff_n / (eff_n + 3.0)
            b_star = w * b_i + (1.0 - w) * b_med
        b_star = min(max(b_n, b_med) + 1.5, max(0.0, b_star))
        e_target = max(e_last, e_last * math.exp(b_star * du)) if e_last > 0 else 0.5
        med = 100.0 * e_target / n_target

        # A trusted saturating fit caps the power law's optimism.
        med_sat = sat[i]
        disagree = 0.0
        if med_sat is not None and med_sat < med:
            disagree = abs(logit(med) - logit(med_sat))
            med = med_sat

        last_rate = last["rates"][i] if i < len(last["rates"]) and last["rates"][i] is not None else None
        if last_rate is not None and last_rate >= 99.99:
            med = 100.0
            spread = S_SATURATED
        else:
            spread = math.sqrt(
                S_BASE ** 2
                + (S_HORIZON * math.sqrt(days)) ** 2
                + min(resid, 0.6) ** 2
                + (0.25 / math.sqrt(max(e_target, 1.0))) ** 2
                + (0.25 * disagree) ** 2
            )
            spread = min(S_MAX, max(S_MIN, spread))
        lg = logit(med)
        qs = [round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z]
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the scored ones")
    trophies = case.get("trophies") or []
    if not trophies:
        raise ValueError("case lists no trophies")
    midnight = release_midnight(case)
    snaps = distinct_snapshots(case, midnight)
    if not snaps:
        # No history at all: a tier prior with a very wide band, so the
        # case still validates instead of crashing the run.
        prior = {"platinum": 0.1, "gold": 0.2, "silver": 0.5, "bronze": 1.0}
        rows = []
        for trophy in trophies:
            med = prior.get(trophy.get("tier"), 1.0)
            lg = logit(med)
            qs = [round4(clamp_pct(inv_logit(lg + z * 1.5))) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}
    if len(snaps) < 3:
        return forecast_flat(case, snaps, midnight)
    return forecast_rich(case, snaps, midnight)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
