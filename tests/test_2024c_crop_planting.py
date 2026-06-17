"""
Comprehensive test for 2024C crop planting strategy problem.
Tests 10 MathFlow modules with simulated agricultural data.
12 plots, 8 crops, 5 years of historical yield/price data.
"""

import sys
import traceback
import numpy as np

np.random.seed(42)

# ============================================================
# DATA GENERATION: 12 plots, 8 crops, 5 years
# ============================================================
N_PLOTS = 12
N_CROPS = 8
N_YEARS = 5

PLOT_NAMES = [f"Plot{i+1}" for i in range(N_PLOTS)]
CROP_NAMES = ["Wheat", "Corn", "Rice", "Soybean", "Cotton", "Vegetable", "Potato", "Peanut"]

# Historical yield: shape (5 years, 12 plots, 8 crops) in kg/mu
base_yields = np.array([
    [300, 450, 500, 150, 200, 1800, 1500, 250],
    [420, 550, 600, 180, 250, 2000, 1800, 300],
    [600, 700, 750, 200, 150, 1200, 2200, 350],
    [280, 400, 480, 160, 300, 2200, 1400, 280],
    [350, 500, 550, 170, 180, 1600, 1600, 260],
    [250, 380, 450, 140, 220, 1900, 1300, 240],
    [380, 520, 580, 190, 270, 2100, 1700, 310],
    [320, 460, 510, 155, 190, 1750, 1550, 255],
    [450, 600, 650, 200, 240, 2050, 1900, 320],
    [270, 390, 470, 145, 210, 1850, 1350, 245],
    [340, 490, 540, 165, 170, 1550, 1650, 270],
    [310, 440, 500, 150, 230, 1950, 1500, 260],
], dtype=float)  # shape (12, 8) = (plots, crops)

# Year variation factors
year_factors = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
# shape (5, 12, 8)
historical_yields = np.array([base_yields * f + np.random.normal(0, 20, base_yields.shape)
                               for f in year_factors])

# Historical prices: shape (5, 8) in yuan/kg
base_prices = np.array([2.5, 2.0, 3.0, 4.5, 8.0, 3.5, 1.8, 6.0])
historical_prices = np.array([base_prices * (1 + 0.05 * (y - 2) + np.random.normal(0, 0.1, N_CROPS))
                               for y in range(N_YEARS)])

# Weather factors: shape (5, 3) -- rainfall, temperature, sunshine
weather = np.array([
    [800, 15.2, 1800],
    [850, 15.8, 1850],
    [900, 16.0, 1900],
    [780, 15.5, 1750],
    [920, 16.3, 1920],
], dtype=float)

# Aggregate yield per year per crop (mean across plots)
agg_yields = historical_yields.mean(axis=1)  # (5, 8)

# For yield series (single crop over years for time-series tests)
wheat_yield_series = agg_yields[:, 0]  # 5 points -- too short for some tests
# Extended series: 20 monthly-like synthetic data points
np.random.seed(123)
extended_yield = 300 + 5 * np.arange(20) + 10 * np.sin(2 * np.pi * np.arange(20) / 4) + np.random.normal(0, 15, 20)

# Planting allocation: 8 crops for LP variables
# Cost per mu for each crop
costs = np.array([400, 500, 600, 350, 700, 800, 450, 550], dtype=float)
# Profit per mu = (yield * price - cost)
profits_per_mu = base_yields.mean(axis=0) * base_prices / 1000 - costs / 1000  # simplified

print("=" * 70)
print("  MathFlow Test: 2024C Crop Planting Strategy")
print("=" * 70)

issues = []
issue_counter = [0]


def report(module, method, error, severity):
    issue_counter[0] += 1
    tag = f"ISSUE #{issue_counter[0]}"
    print(f"  {tag}: [{module}.{method}] | ERROR: {error} | SEVERITY: {severity}")
    issues.append((tag, module, method, error, severity))


# ============================================================
# TEST 1: GreyPrediction -- forecast crop yields
# ============================================================
print("\n--- Test 1: GreyPrediction (predict.GreyPrediction) ---")
try:
    from mathflow.predict import GreyPrediction

    # Use 5 years of wheat yield (mean across plots) -- minimal data for GM(1,1)
    wheat_data = agg_yields[:, 0]
    print(f"  Input data (wheat yield, 5 years): {wheat_data}")

    gp = GreyPrediction(wheat_data)
    gp.fit()
    r = gp.result
    print(f"  GM(1,1) a={r.a:.4f}, b={r.b:.4f}, C={r.posterior_ratio:.4f}, P={r.small_error_prob:.4f}")
    print(f"  Accuracy: {r.accuracy_level}")

    forecast = gp.predict(steps=3)
    print(f"  Forecast (3 steps): {forecast}")

    # Check forecast length
    assert len(forecast) == 3, f"Expected 3 forecasts, got {len(forecast)}"
    # Check fitted values length
    assert len(r.fitted_values) == len(wheat_data), "Fitted values length mismatch"

    # Test summary
    s = gp.summary(steps=3)
    assert "GM(1,1)" in s, "Summary missing GM(1,1) header"

    # Test predict before fit
    try:
        gp2 = GreyPrediction([1, 2, 3, 4])
        gp2.predict(steps=1)
        report("predict.GreyPrediction", "predict", "No RuntimeError raised when predict() called before fit()", "high")
    except RuntimeError:
        pass  # expected

    # Test with too few data points
    try:
        gp3 = GreyPrediction([1, 2, 3])
        report("predict.GreyPrediction", "__init__", "No ValueError raised for < 4 data points", "high")
    except ValueError:
        pass  # expected

    print("  [PASS] GreyPrediction")

except Exception as e:
    report("predict.GreyPrediction", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 2: ARIMAModel -- auto_fit on price series
# ============================================================
print("\n--- Test 2: ARIMAModel (predict.ARIMAModel) ---")
try:
    from mathflow.predict import ARIMAModel

    # Use extended yield series (20 points) for ARIMA
    arima_data = extended_yield
    print(f"  Input data (20 points): {arima_data[:5]}... (len={len(arima_data)})")

    arima = ARIMAModel(arima_data)
    arima.auto_fit(max_p=2, max_d=1, max_q=2, criterion="aic")
    print(f"  Best order: {arima.order}")
    print(f"  AIC: {arima.aic:.2f}")

    forecast = arima.predict(steps=5)
    print(f"  Forecast (5 steps): {forecast}")
    assert len(forecast) == 5, f"Expected 5 forecasts, got {len(forecast)}"

    # Confidence interval
    ci = arima.get_confidence_interval(steps=5)
    print(f"  CI shape: {ci.shape}")

    # Test predict before fit
    try:
        arima2 = ARIMAModel(arima_data)
        arima2.predict(steps=1)
        report("predict.ARIMAModel", "predict", "No RuntimeError raised when predict() called before fit()/auto_fit()", "high")
    except RuntimeError:
        pass

    # Test with very short data (stress case)
    try:
        arima3 = ARIMAModel([1.0, 2.0, 3.0])
        arima3.auto_fit(max_p=1, max_d=1, max_q=1)
        report("predict.ARIMAModel", "auto_fit", "auto_fit did not raise on 3-point data (may produce garbage)", "medium")
    except Exception as e:
        print(f"  auto_fit with 3 points raised: {type(e).__name__}: {e} -- acceptable")

    # Test summary
    s = arima.summary()
    assert isinstance(s, str) and len(s) > 0, "summary() returned empty"
    print("  [PASS] ARIMAModel")

except Exception as e:
    report("predict.ARIMAModel", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 3: TimeSeriesDecompose -- decompose yield series
# ============================================================
print("\n--- Test 3: TimeSeriesDecompose (timeseries.TimeSeriesDecompose) ---")
try:
    from mathflow.timeseries import TimeSeriesDecompose

    # Use extended yield (20 points), period=4 (quarterly)
    ts = TimeSeriesDecompose(extended_yield, period=4, model="additive")
    result = ts.decompose(method="moving_average")
    print(f"  Trend range: [{result.trend.min():.2f}, {result.trend.max():.2f}]")
    print(f"  Seasonal range: [{result.seasonal.min():.2f}, {result.seasonal.max():.2f}]")
    print(f"  Residual std: {result.residual.std():.4f}")
    assert len(result.trend) == len(extended_yield), "Trend length mismatch"
    assert len(result.seasonal) == len(extended_yield), "Seasonal length mismatch"
    assert len(result.residual) == len(extended_yield), "Residual length mismatch"

    # Test STL method
    ts2 = TimeSeriesDecompose(extended_yield, period=4, model="additive")
    result2 = ts2.decompose(method="stl")
    print(f"  STL trend range: [{result2.trend.min():.2f}, {result2.trend.max():.2f}]")
    assert len(result2.trend) == len(extended_yield), "STL trend length mismatch"

    # Test multiplicative model
    # Multiplicative needs positive data
    ts3 = TimeSeriesDecompose(extended_yield, period=4, model="multiplicative")
    result3 = ts3.decompose(method="moving_average")
    print(f"  Multiplicative seasonal range: [{result3.seasonal.min():.4f}, {result3.seasonal.max():.4f}]")

    # Test invalid method
    try:
        ts4 = TimeSeriesDecompose(extended_yield, period=4)
        ts4.decompose(method="invalid_method")
        report("timeseries.TimeSeriesDecompose", "decompose", "No ValueError raised for invalid method", "medium")
    except ValueError:
        pass

    # Test summary
    s = ts.summary()
    assert "时间序列分解" in s, "Summary missing header"
    print("  [PASS] TimeSeriesDecompose")

except Exception as e:
    report("timeseries.TimeSeriesDecompose", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 4: MultiRegression -- yield vs weather factors
# ============================================================
print("\n--- Test 4: MultiRegression (stats.MultiRegression) ---")
try:
    from mathflow.stats import MultiRegression

    # Regress wheat yield on weather factors (5 data points, 3 features)
    # This is an extreme low-sample regression
    X_weather = weather  # (5, 3)
    y_wheat = agg_yields[:, 0]  # (5,)
    print(f"  X shape: {X_weather.shape}, y shape: {y_wheat.shape}")

    mr = MultiRegression(X_weather, y_wheat, var_names=["Rainfall", "Temperature", "Sunshine"])
    mr.fit()
    r = mr.result
    print(f"  R2={r.r2:.4f}, adj_R2={r.adj_r2:.4f}")
    print(f"  Coefficients: {r.coefficients}")
    print(f"  Intercept: {r.intercept:.4f}")
    print(f"  F-stat: {r.f_statistic:.4f}, F-p: {r.f_p_value:.4e}")

    # Predict
    X_new = np.array([[850, 16.0, 1850]])
    y_pred = mr.predict(X_new)
    print(f"  Prediction for {X_new}: {y_pred}")
    assert len(y_pred) == 1, f"Expected 1 prediction, got {len(y_pred)}"

    # Test with perfect collinearity (should still work via lstsq)
    X_collinear = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]], dtype=float)
    try:
        mr2 = MultiRegression(X_collinear, y_wheat)
        mr2.fit()
        print(f"  Collinear case R2={mr2.result.r2:.4f} (expected near 1.0 or degenerate)")
    except Exception as e:
        print(f"  Collinear case raised: {type(e).__name__}: {e}")

    # Test summary
    s = mr.summary()
    assert "多元线性回归" in s, "Summary missing header"

    # Test predict before fit
    try:
        mr3 = MultiRegression(X_weather, y_wheat)
        mr3.predict(X_new)
        report("stats.MultiRegression", "predict", "No RuntimeError raised when predict() called before fit()", "high")
    except RuntimeError:
        pass

    print("  [PASS] MultiRegression")

except Exception as e:
    report("stats.MultiRegression", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 5: LinearProgramming -- optimize planting allocation
# ============================================================
print("\n--- Test 5: LinearProgramming (optimize.LinearProgramming) ---")
try:
    from mathflow.optimize import LinearProgramming

    # Maximize profit from planting 8 crops
    # Profit coefficients (simplified per-mu profit)
    profit_coeffs = (base_yields.mean(axis=0) * base_prices / 1000).tolist()
    print(f"  Profit coefficients: {[f'{p:.2f}' for p in profit_coeffs]}")

    lp = LinearProgramming()
    lp.set_objective(profit_coeffs, sense="max", var_names=CROP_NAMES)

    # Total land <= 1200 mu (100 mu per plot)
    total_land = [1] * N_CROPS
    lp.add_constraint(total_land, "<=", 1200.0)

    # Each crop at least 50 mu (diversity constraint)
    for i in range(N_CROPS):
        coeffs = [0] * N_CROPS
        coeffs[i] = 1
        lp.add_constraint(coeffs, ">=", 50.0)

    # Water constraint: sum of water_use * area <= budget
    water_use = [200, 300, 400, 150, 250, 180, 220, 160]
    lp.add_constraint(water_use, "<=", 200000.0)

    result = lp.solve()
    print(f"  Status: {result.status}")
    print(f"  Optimal value: {result.optimal_value:.4f}")
    print(f"  Solution: {result.solution}")
    assert result.status == "Optimal", f"Expected 'Optimal', got '{result.status}'"
    assert len(result.solution) == N_CROPS, "Solution length mismatch"

    # Check constraints are satisfied
    total = sum(result.solution)
    print(f"  Total land used: {total:.2f} (constraint: <= 1200)")
    assert total <= 1200 + 1e-6, f"Land constraint violated: {total}"

    for i, v in enumerate(result.solution):
        if v < 50 - 1e-6:
            report("optimize.LinearProgramming", "solve", f"Minimum area constraint violated for {CROP_NAMES[i]}: {v:.2f} < 50", "high")

    water_total = sum(w * a for w, a in zip(water_use, result.solution))
    print(f"  Water used: {water_total:.2f} (constraint: <= 200000)")
    assert water_total <= 200000 + 1e-6, f"Water constraint violated: {water_total}"

    # Test summary
    s = lp.summary()
    assert "线性规划" in s, "Summary missing header"
    print("  [PASS] LinearProgramming")

except Exception as e:
    report("optimize.LinearProgramming", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 6: GeneticAlgorithm -- GA for nonlinear allocation
# ============================================================
print("\n--- Test 6: GeneticAlgorithm (optimize.GeneticAlgorithm) ---")
try:
    from mathflow.optimize import GeneticAlgorithm

    # Nonlinear profit function with risk penalty
    # Maximize: sum of (profit_i * x_i) - risk * variance
    profit_arr = base_yields.mean(axis=0) * base_prices / 1000
    risk_aversion = 0.01

    def fitness(x):
        # x = allocation for 8 crops (mu)
        revenue = np.sum(profit_arr * x)
        risk_penalty = risk_aversion * np.var(x)
        # Penalty for violating total land constraint
        total = np.sum(x)
        penalty = max(0, total - 1200) * 10
        # Penalty for negative values
        penalty += np.sum(np.maximum(0, -x)) * 100
        return revenue - risk_penalty - penalty

    bounds = [(50, 200)] * N_CROPS  # each crop 50-200 mu

    ga = GeneticAlgorithm(
        fitness_func=fitness,
        n_vars=N_CROPS,
        bounds=bounds,
        pop_size=80,
        generations=150,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elitism=2,
        encoding="real",
    )
    result = ga.run(verbose=False)
    print(f"  Best fitness: {result.best_fitness:.4f}")
    print(f"  Best solution: {result.best_solution}")
    print(f"  Convergence gen: {result.convergence_generation}")

    assert len(result.best_solution) == N_CROPS, "Solution length mismatch"
    assert result.best_fitness > -1e10, "Fitness is -inf (degenerate)"
    assert "best" in result.history and "mean" in result.history and "worst" in result.history, \
        "History missing expected keys"

    # Check bounds
    for i in range(N_CROPS):
        if result.best_solution[i] < bounds[i][0] - 1e-6 or result.best_solution[i] > bounds[i][1] + 1e-6:
            report("optimize.GeneticAlgorithm", "run", f"Variable {i} out of bounds: {result.best_solution[i]:.2f}", "medium")

    # Test summary
    s = ga.summary()
    assert "遗传算法" in s, "Summary missing header"

    # Test run before... well, GA doesn't have a fit/run separation issue.
    # Test with degenerate fitness (constant)
    try:
        ga2 = GeneticAlgorithm(fitness_func=lambda x: 42.0, n_vars=2, bounds=[(0, 1), (0, 1)],
                               pop_size=10, generations=5)
        r2 = ga2.run()
        print(f"  Constant fitness test: best={r2.best_fitness:.1f}")
    except Exception as e:
        report("optimize.GeneticAlgorithm", "run", f"Constant fitness raised: {type(e).__name__}: {e}", "medium")

    print("  [PASS] GeneticAlgorithm")

except Exception as e:
    report("optimize.GeneticAlgorithm", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 7: EntropyWeight -- weight determination
# ============================================================
print("\n--- Test 7: EntropyWeight (evaluate.EntropyWeight) ---")
try:
    from mathflow.evaluate import EntropyWeight

    # Evaluation matrix: 8 plans (crops), 4 indicators
    # Indicators: yield, price, cost, risk
    eval_data = np.array([
        [300, 2.5, 400, 0.15],
        [450, 2.0, 500, 0.20],
        [500, 3.0, 600, 0.18],
        [150, 4.5, 350, 0.25],
        [200, 8.0, 700, 0.30],
        [1800, 3.5, 800, 0.12],
        [1500, 1.8, 450, 0.22],
        [250, 6.0, 550, 0.20],
    ], dtype=float)

    # Types: yield=benefit(1), price=benefit(1), cost=cost(-1), risk=cost(-1)
    ew = EntropyWeight(eval_data, types=[1, 1, -1, -1])
    ew.fit()
    w = ew.weights
    print(f"  Weights: {w}")
    print(f"  Sum of weights: {w.sum():.6f}")
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {w.sum()}"
    assert all(wi >= 0 for wi in w), "Negative weight detected"

    r = ew.result
    print(f"  Entropy values: {r.entropy_values}")
    print(f"  Redundancy: {r.redundancy}")
    print(f"  Variation: {r.variation}")

    # Test without types
    ew2 = EntropyWeight(eval_data)
    ew2.fit()
    w2 = ew2.weights
    print(f"  Weights (no types): {w2}")
    assert abs(w2.sum() - 1.0) < 1e-6, "Weights without types don't sum to 1"

    # Test with uniform data (all same -> entropy=1, weight=uniform)
    uniform_data = np.ones((5, 3)) * 100
    ew3 = EntropyWeight(uniform_data)
    ew3.fit()
    w3 = ew3.weights
    print(f"  Uniform data weights: {w3} (should be ~equal)")
    assert all(abs(wi - 1/3) < 0.1 for wi in w3), "Uniform data should yield ~equal weights"

    # Test 1D data (should raise ValueError)
    try:
        ew4 = EntropyWeight([1, 2, 3, 4])
        ew4.fit()
        report("evaluate.EntropyWeight", "__init__", "No ValueError raised for 1D data input", "high")
    except ValueError:
        pass

    # Test summary
    s = ew.summary(labels=CROP_NAMES[:4])
    assert "熵权法" in s, "Summary missing header"
    print("  [PASS] EntropyWeight")

except Exception as e:
    report("evaluate.EntropyWeight", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 8: TOPSIS -- rank planting plans
# ============================================================
print("\n--- Test 8: TOPSIS (evaluate.TOPSIS) ---")
try:
    from mathflow.evaluate import TOPSIS

    # Use same evaluation data, 8 plans, 4 indicators
    # Types: yield=benefit, price=benefit, cost=cost, risk=cost
    topsis = TOPSIS(eval_data, weights=[0.3, 0.25, 0.2, 0.25], types=[1, 1, -1, -1])
    topsis.fit()

    scores = topsis.scores
    rankings = topsis.rankings
    print(f"  Scores: {scores}")
    print(f"  Rankings: {rankings}")

    assert len(scores) == 8, f"Expected 8 scores, got {len(scores)}"
    assert len(rankings) == 8, f"Expected 8 rankings, got {len(rankings)}"
    assert all(0 <= s <= 1 for s in scores), "Score out of [0,1] range"
    assert set(rankings) == set(range(1, 9)), "Rankings should be 1-8"

    r = topsis.result
    print(f"  Ideal best: {r.ideal_best}")
    print(f"  Ideal worst: {r.ideal_worst}")
    print(f"  Dist best: {r.dist_best}")

    # Test without weights (equal weights)
    topsis2 = TOPSIS(eval_data, types=[1, 1, -1, -1])
    topsis2.fit()
    print(f"  Scores (equal weights): {topsis2.scores}")

    # Test without types (default all benefit)
    topsis3 = TOPSIS(eval_data, weights=[0.3, 0.25, 0.2, 0.25])
    topsis3.fit()
    print(f"  Scores (default types): {topsis3.scores}")

    # Test 1D data (should raise)
    try:
        topsis4 = TOPSIS([1, 2, 3])
        report("evaluate.TOPSIS", "__init__", "No ValueError raised for 1D data input", "high")
    except ValueError:
        pass

    # Test with zero column (edge case)
    zero_col_data = eval_data.copy()
    zero_col_data[:, 0] = 0
    try:
        topsis5 = TOPSIS(zero_col_data, types=[1, 1, -1, -1])
        topsis5.fit()
        print(f"  Zero-column test scores: {topsis5.scores}")
    except Exception as e:
        report("evaluate.TOPSIS", "fit", f"Zero-column data raised: {type(e).__name__}: {e}", "medium")

    # Test summary
    s = topsis.summary(labels=CROP_NAMES)
    assert "TOPSIS" in s, "Summary missing header"
    print("  [PASS] TOPSIS")

except Exception as e:
    report("evaluate.TOPSIS", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 9: DistributionFitter -- fit yield distributions
# ============================================================
print("\n--- Test 9: DistributionFitter (prob.DistributionFitter) ---")
try:
    from mathflow.prob import DistributionFitter

    # Use historical wheat yields across all plots and years as sample
    wheat_all = historical_yields[:, :, 0].flatten()  # 60 data points
    print(f"  Sample size: {len(wheat_all)}, mean={wheat_all.mean():.2f}, std={wheat_all.std():.2f}")

    df = DistributionFitter(wheat_all)
    results = df.auto_fit(top_n=3)
    print(f"  Top 3 distributions by AIC:")
    for r in results:
        print(f"    {r.distribution}: AIC={r.aic:.2f}, BIC={r.bic:.2f}, KS-p={r.ks_p_value:.4f}")

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0].aic <= results[1].aic <= results[2].aic, "Results not sorted by AIC"

    # Test specific distribution fit
    norm_result = df.fit("norm")
    print(f"  Normal fit: params={norm_result.params}, KS-p={norm_result.ks_p_value:.4f}")
    assert len(norm_result.params) == 2, "Normal should have 2 params (loc, scale)"

    # Test goodness of fit
    gof = df.goodness_of_fit()
    print(f"  Goodness of fit ({len(gof)} distributions tested):")
    for name, info in gof.items():
        print(f"    {name}: KS-p={info['K-S p值']:.4f}, pass={info['通过(p>0.05)']}")

    # Test unknown distribution
    try:
        df.fit("unknown_dist")
        report("prob.DistributionFitter", "fit", "No ValueError raised for unknown distribution name", "high")
    except ValueError:
        pass

    # Test summary
    s = df.summary()
    assert "概率分布拟合" in s, "Summary missing header"

    # Test with small sample
    try:
        df_small = DistributionFitter([1.0, 2.0, 3.0])
        r_small = df_small.auto_fit()
        print(f"  Small sample auto_fit: {len(r_small)} results")
    except Exception as e:
        report("prob.DistributionFitter", "auto_fit", f"Small sample raised: {type(e).__name__}: {e}", "medium")

    print("  [PASS] DistributionFitter")

except Exception as e:
    report("prob.DistributionFitter", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# TEST 10: FullPaper -- generate full paper
# ============================================================
print("\n--- Test 10: FullPaper (paper.FullPaper) ---")
try:
    from mathflow.paper import FullPaper

    paper = FullPaper("2024年C题 农作物种植策略优化", year=2024)
    paper.set_background(
        "某乡村现有12块耕地，需要在满足作物种植需求的前提下，"
        "制定2024-2026年的最优种植方案，以最大化种植收益。"
    )
    paper.set_data_source("附件1：乡村种植数据.xlsx")

    paper.add_assumption("假设各耕地土壤条件在规划期内保持稳定")
    paper.add_assumption("假设农作物市场价格波动服从正态分布")
    paper.add_assumption("假设农作物产量与天气因素存在线性关系")
    paper.add_assumption("假设同一地块同一季节只能种植一种作物")

    paper.add_symbol("x_{ij}", "第i块地第j种作物的种植面积", "亩")
    paper.add_symbol("P_j", "第j种作物的单位售价", "元/kg")
    paper.add_symbol("Y_{ij}", "第i块地第j种作物的亩产量", "kg/亩")
    paper.add_symbol("C_j", "第j种作物的种植成本", "元/亩")
    paper.add_symbol("Z", "总收益", "元")

    # Sub-problem 1
    paper.add_sub_problem(
        description="制定2024年最优种植方案",
        method="线性规划模型",
        result="最优方案总收益为XX万元",
        model_name="线性规划模型",
    )

    # Sub-problem 2
    paper.add_sub_problem(
        description="考虑未来不确定性，制定2024-2026年三年种植方案",
        method="遗传算法 + 灰色预测",
        result="三年最优方案总收益为XX万元",
        model_name="多目标优化模型",
    )

    # Sub-problem 3
    paper.add_sub_problem(
        description="风险分析与鲁棒性检验",
        method="蒙特卡洛模拟 + 灵敏度分析",
        result="模型具有较好的鲁棒性",
    )

    paper.add_sensitivity_param("亩产量", 500.0, "主要农作物平均亩产量")
    paper.add_sensitivity_param("市场价格", 3.0, "农作物平均市场价格")
    paper.add_sensitivity_param("种植成本", 500.0, "农作物平均种植成本")

    paper.set_keywords(["农作物种植", "线性规划", "遗传算法", "灰色预测", "TOPSIS"])

    # Generate
    latex_content = paper.generate("/tmp/test_crop_paper.tex")
    print(f"  Generated LaTeX length: {len(latex_content)} chars")
    assert len(latex_content) > 1000, "LaTeX content too short"
    assert "\\documentclass" in latex_content, "Missing documentclass"
    assert "\\begin{document}" in latex_content, "Missing begin document"
    assert "\\end{document}" in latex_content, "Missing end document"
    assert "\\begin{abstract}" in latex_content, "Missing abstract"
    assert "问题1" in latex_content, "Missing sub-problem 1 section"
    assert "问题2" in latex_content, "Missing sub-problem 2 section"
    assert "问题3" in latex_content, "Missing sub-problem 3 section"
    assert "灵敏度分析" in latex_content, "Missing sensitivity section"
    assert "模型评价" in latex_content, "Missing evaluation section"
    assert "参考文献" in latex_content, "Missing references section"
    assert "附录" in latex_content, "Missing appendix"

    # Verify file was saved
    import os
    assert os.path.exists("/tmp/test_crop_paper.tex"), "File not saved"

    # Test with no sub-problems (edge case)
    try:
        paper2 = FullPaper("Test Empty Paper")
        latex2 = paper2.generate()
        assert "\\documentclass" in latex2, "Empty paper missing documentclass"
        print("  Empty paper generation: OK")
    except Exception as e:
        report("paper.FullPaper", "generate", f"Empty paper raised: {type(e).__name__}: {e}", "medium")

    # Test add_evaluator with ModelEvaluator
    try:
        from mathflow.paper import ModelEvaluator
        ev = ModelEvaluator("种植策略优化模型")
        ev.add_strength("模型考虑了多种约束条件")
        ev.add_strength("采用多种方法交叉验证")
        ev.add_weakness("历史数据量有限")
        ev.set_metric("R²", "0.95")
        paper.add_evaluator(ev)
        latex3 = paper.generate()
        assert "模型考虑了多种约束条件" in latex3, "Evaluator strengths not in output"
        print("  Evaluator integration: OK")
    except Exception as e:
        report("paper.FullPaper", "add_evaluator", f"ModelEvaluator integration raised: {type(e).__name__}: {e}", "medium")

    print("  [PASS] FullPaper")

except Exception as e:
    report("paper.FullPaper", "overall", f"{type(e).__name__}: {e}", "critical")
    traceback.print_exc()


# ============================================================
# INTEGRATION TEST: Combined workflow
# ============================================================
print("\n--- Integration Test: Combined workflow ---")
try:
    # 1. Grey predict yields
    from mathflow.predict import GreyPrediction
    gp = GreyPrediction(agg_yields[:, 0])
    gp.fit()
    future_yield = gp.predict(steps=1)[0]
    print(f"  Predicted future wheat yield: {future_yield:.2f}")

    # 2. EntropyWeight + TOPSIS for ranking
    from mathflow.evaluate import EntropyWeight, TOPSIS
    ew = EntropyWeight(eval_data, types=[1, 1, -1, -1])
    ew.fit()
    weights = ew.weights
    print(f"  Entropy weights: {weights}")

    topsis = TOPSIS(eval_data, weights=weights, types=[1, 1, -1, -1])
    topsis.fit()
    print(f"  TOPSIS rankings: {topsis.rankings}")
    best_crop_idx = np.argmin(topsis.rankings)
    print(f"  Best crop by TOPSIS: {CROP_NAMES[best_crop_idx]}")

    # 3. LP with predicted data
    from mathflow.optimize import LinearProgramming
    lp = LinearProgramming()
    predicted_profits = base_yields.mean(axis=0) * base_prices / 1000
    lp.set_objective(predicted_profits.tolist(), sense="max", var_names=CROP_NAMES)
    lp.add_constraint([1]*N_CROPS, "<=", 1200.0)
    for i in range(N_CROPS):
        c = [0]*N_CROPS
        c[i] = 1
        lp.add_constraint(c, ">=", 50.0)
    lp_result = lp.solve()
    print(f"  LP optimal profit: {lp_result.optimal_value:.2f}")

    # 4. Generate paper with all results
    from mathflow.paper import FullPaper
    paper = FullPaper("2024年C题 农作物种植策略优化", year=2024)
    paper.add_sub_problem("最优种植方案", "线性规划",
                          result=f"最优收益{lp_result.optimal_value:.2f}万元")
    paper.add_sub_problem("方案评价", "熵权法+TOPSIS",
                          result=f"最优作物为{CROP_NAMES[best_crop_idx]}")
    paper.add_sensitivity_param("亩产量", float(future_yield))
    latex = paper.generate("/tmp/test_integration.tex")
    assert len(latex) > 500, "Integration paper too short"
    print("  Integration paper generated successfully")
    print("  [PASS] Integration Test")

except Exception as e:
    report("integration", "workflow", f"{type(e).__name__}: {e}", "high")
    traceback.print_exc()


# ============================================================
# EDGE CASE TESTS: Stress and boundary conditions
# ============================================================
print("\n--- Edge Case Tests ---")

# Edge 1: GreyPrediction with constant data (a should be ~0)
try:
    from mathflow.predict import GreyPrediction
    gp_const = GreyPrediction([100, 100, 100, 100, 100])
    gp_const.fit()
    r = gp_const.result
    forecast = gp_const.predict(steps=2)
    print(f"  GreyPrediction constant: a={r.a:.6f}, forecast={forecast}")
    if any(np.isnan(forecast)) or any(np.isinf(forecast)):
        report("predict.GreyPrediction", "predict", "NaN/Inf in forecast for constant data", "high")
    else:
        print("  GreyPrediction constant: OK")
except Exception as e:
    report("predict.GreyPrediction", "fit/predict", f"Constant data raised: {type(e).__name__}: {e}", "high")

# Edge 2: GreyPrediction with rapidly growing data (may cause numerical issues)
try:
    gp_exp = GreyPrediction([1, 2, 4, 8, 16, 32, 64])
    gp_exp.fit()
    r = gp_exp.result
    forecast = gp_exp.predict(steps=3)
    print(f"  GreyPrediction exponential: a={r.a:.4f}, forecast={forecast}")
    if any(np.isnan(forecast)) or any(np.isinf(forecast)):
        report("predict.GreyPrediction", "predict", "NaN/Inf in forecast for exponential data", "high")
    else:
        print("  GreyPrediction exponential: OK")
except Exception as e:
    report("predict.GreyPrediction", "fit/predict", f"Exponential data raised: {type(e).__name__}: {e}", "high")

# Edge 3: ARIMAModel with constant data
try:
    from mathflow.predict import ARIMAModel
    const_series = [50.0] * 20
    arima_const = ARIMAModel(const_series)
    arima_const.fit(order=(1, 0, 0))
    forecast = arima_const.predict(steps=3)
    print(f"  ARIMA constant: forecast={forecast}")
    if any(np.isnan(forecast)) or any(np.isinf(forecast)):
        report("predict.ARIMAModel", "predict", "NaN/Inf in forecast for constant data", "high")
    else:
        print("  ARIMA constant: OK")
except Exception as e:
    report("predict.ARIMAModel", "fit/predict", f"Constant data raised: {type(e).__name__}: {e}", "high")

# Edge 4: ARIMAModel auto_fit with very short data (5 points)
try:
    short_arima = ARIMAModel([10.0, 12.0, 11.0, 13.0, 14.0])
    short_arima.auto_fit(max_p=1, max_d=1, max_q=1)
    forecast = short_arima.predict(steps=2)
    print(f"  ARIMA 5-point: order={short_arima.order}, forecast={forecast}")
    print("  ARIMA 5-point auto_fit: OK")
except ValueError as e:
    # 正确的边界检查行为：数据点不足应该抛出 ValueError
    print(f"  ARIMA 5-point: raised ValueError (correct validation)")
except Exception as e:
    report("predict.ARIMAModel", "auto_fit", f"5-point data raised unexpected: {type(e).__name__}: {e}", "medium")

# Edge 5: MultiRegression with more features than samples (p > n)
try:
    from mathflow.stats import MultiRegression
    X_tall = np.random.randn(3, 5)  # 3 samples, 5 features
    y_tall = np.random.randn(3)
    mr_tall = MultiRegression(X_tall, y_tall)
    mr_tall.fit()
    r = mr_tall.result
    print(f"  MultiRegression p>n: R2={r.r2:.4f} (expected: overfit)")
    if np.isnan(r.r2) or np.isinf(r.r2):
        report("stats.MultiRegression", "fit", "NaN/Inf R2 when p > n", "high")
    else:
        print("  MultiRegression p>n: OK")
except Exception as e:
    report("stats.MultiRegression", "fit", f"p>n case raised: {type(e).__name__}: {e}", "high")

# Edge 6: MultiRegression with add_constant=False
try:
    X_nc = np.random.randn(20, 3)
    y_nc = 2 * X_nc[:, 0] + 3 * X_nc[:, 1] + np.random.randn(20) * 0.1
    mr_nc = MultiRegression(X_nc, y_nc, add_constant=False)
    mr_nc.fit()
    r = mr_nc.result
    print(f"  MultiRegression no constant: R2={r.r2:.4f}, intercept={r.intercept}")
    if r.intercept != 0.0:
        report("stats.MultiRegression", "fit", f"add_constant=False but intercept={r.intercept} (expected 0.0)", "medium")
    else:
        print("  MultiRegression no constant: OK")
except Exception as e:
    report("stats.MultiRegression", "fit", f"add_constant=False raised: {type(e).__name__}: {e}", "medium")

# Edge 7: EntropyWeight with negative values
try:
    from mathflow.evaluate import EntropyWeight
    neg_data = np.array([[1, -2, 3], [4, -5, 6], [7, -8, 9]], dtype=float)
    ew_neg = EntropyWeight(neg_data)
    ew_neg.fit()
    w = ew_neg.weights
    print(f"  EntropyWeight negative data: weights={w}")
    if any(np.isnan(w)) or any(np.isinf(w)):
        report("evaluate.EntropyWeight", "fit", "NaN/Inf weights with negative data", "high")
    elif any(w < 0):
        report("evaluate.EntropyWeight", "fit", f"Negative weight detected: {w}", "high")
    else:
        print("  EntropyWeight negative data: OK")
except Exception as e:
    report("evaluate.EntropyWeight", "fit", f"Negative data raised: {type(e).__name__}: {e}", "medium")

# Edge 8: EntropyWeight with zero data
try:
    zero_data = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]], dtype=float)
    ew_zero = EntropyWeight(zero_data)
    ew_zero.fit()
    w = ew_zero.weights
    print(f"  EntropyWeight zero column: weights={w}")
    if any(np.isnan(w)):
        report("evaluate.EntropyWeight", "fit", "NaN weights with zero column", "high")
    else:
        print("  EntropyWeight zero column: OK")
except Exception as e:
    report("evaluate.EntropyWeight", "fit", f"Zero column data raised: {type(e).__name__}: {e}", "medium")

# Edge 9: TOPSIS with identical rows
try:
    from mathflow.evaluate import TOPSIS
    same_data = np.array([[10, 20, 30], [10, 20, 30], [10, 20, 30]], dtype=float)
    topsis_same = TOPSIS(same_data)
    topsis_same.fit()
    scores = topsis_same.scores
    rankings = topsis_same.rankings
    print(f"  TOPSIS identical rows: scores={scores}, rankings={rankings}")
    if any(np.isnan(scores)):
        report("evaluate.TOPSIS", "fit", "NaN scores with identical rows", "high")
    else:
        print("  TOPSIS identical rows: OK")
except Exception as e:
    report("evaluate.TOPSIS", "fit", f"Identical rows raised: {type(e).__name__}: {e}", "medium")

# Edge 10: TOPSIS with single row
try:
    single_data = np.array([[10, 20, 30]], dtype=float)
    topsis_single = TOPSIS(single_data)
    topsis_single.fit()
    print(f"  TOPSIS single row: scores={topsis_single.scores}")
    print("  TOPSIS single row: OK")
except Exception as e:
    report("evaluate.TOPSIS", "fit", f"Single row raised: {type(e).__name__}: {e}", "medium")

# Edge 11: DistributionFitter with all-same data
try:
    from mathflow.prob import DistributionFitter
    same_vals = np.ones(50) * 42.0
    df_same = DistributionFitter(same_vals)
    results = df_same.auto_fit()
    print(f"  DistributionFitter constant: {len(results)} results")
    if any(np.isnan(r.aic) for r in results if hasattr(r, 'aic')):
        report("prob.DistributionFitter", "auto_fit", "NaN AIC with constant data", "medium")
    else:
        print("  DistributionFitter constant: OK")
except Exception as e:
    report("prob.DistributionFitter", "auto_fit", f"Constant data raised: {type(e).__name__}: {e}", "medium")

# Edge 12: DistributionFitter with very small sample (4 points)
try:
    small_data = np.array([1.1, 2.2, 3.3, 4.4])
    df_tiny = DistributionFitter(small_data)
    results = df_tiny.auto_fit(top_n=2)
    print(f"  DistributionFitter 4-point: {len(results)} results")
    for r in results:
        if np.isinf(r.aic):
            report("prob.DistributionFitter", "auto_fit", f"Inf AIC for {r.distribution} with 4-point data", "low")
    print("  DistributionFitter 4-point: OK")
except Exception as e:
    report("prob.DistributionFitter", "auto_fit", f"4-point data raised: {type(e).__name__}: {e}", "medium")

# Edge 13: LinearProgramming with infeasible constraints
try:
    from mathflow.optimize import LinearProgramming
    lp_inf = LinearProgramming()
    lp_inf.set_objective([1, 1], sense="max")
    lp_inf.add_constraint([1, 0], ">=", 100)
    lp_inf.add_constraint([1, 0], "<=", 10)
    result = lp_inf.solve()
    print(f"  LP infeasible: status={result.status}")
    if result.status == "Optimal":
        report("optimize.LinearProgramming", "solve", f"Infeasible LP returned status 'Optimal': solution={result.solution}", "high")
    else:
        print(f"  LP infeasible correctly returned: {result.status}")
except Exception as e:
    report("optimize.LinearProgramming", "solve", f"Infeasible LP raised: {type(e).__name__}: {e}", "medium")

# Edge 14: LinearProgramming with unbounded objective
try:
    lp_unb = LinearProgramming()
    lp_unb.set_objective([1, 1], sense="max")
    lp_unb.add_constraint([1, -1], ">=", 0)  # x1 >= x2, but no upper bound
    result = lp_unb.solve()
    print(f"  LP unbounded: status={result.status}, optimal={result.optimal_value}")
    if result.status == "Optimal" and result.optimal_value > 1e10:
        report("optimize.LinearProgramming", "solve", f"Unbounded LP returned finite 'Optimal': {result.optimal_value}", "medium")
    elif result.status == "Unbounded":
        print("  LP unbounded correctly detected")
    else:
        print(f"  LP unbounded: status={result.status}")
except Exception as e:
    report("optimize.LinearProgramming", "solve", f"Unbounded LP raised: {type(e).__name__}: {e}", "medium")

# Edge 15: GeneticAlgorithm with single variable, tight bounds
try:
    from mathflow.optimize import GeneticAlgorithm
    ga_tight = GeneticAlgorithm(
        fitness_func=lambda x: -(x[0] - 0.5)**2,
        n_vars=1,
        bounds=[(0.49, 0.51)],
        pop_size=20,
        generations=50,
    )
    result = ga_tight.run()
    print(f"  GA tight bounds: best={result.best_solution[0]:.6f}, fitness={result.best_fitness:.6f}")
    if abs(result.best_solution[0] - 0.5) > 0.02:
        report("optimize.GeneticAlgorithm", "run", f"Tight bounds solution off: {result.best_solution[0]:.4f} vs expected 0.5", "low")
    else:
        print("  GA tight bounds: OK")
except Exception as e:
    report("optimize.GeneticAlgorithm", "run", f"Tight bounds raised: {type(e).__name__}: {e}", "medium")

# Edge 16: TimeSeriesDecompose with period > data length
try:
    from mathflow.timeseries import TimeSeriesDecompose
    ts_short = TimeSeriesDecompose([1, 2, 3, 4, 5], period=12, model="additive")
    result_short = ts_short.decompose()
    print(f"  TimeSeriesDecompose period>len: trend={result_short.trend}")
    if any(np.isnan(result_short.trend)):
        report("timeseries.TimeSeriesDecompose", "decompose", "NaN in trend when period > data length", "high")
    else:
        print("  TimeSeriesDecompose period>len: OK")
except ValueError as e:
    # 正确的边界检查行为：period > len 应该抛出 ValueError
    print(f"  TimeSeriesDecompose period>len: raised ValueError (correct validation)")
except Exception as e:
    report("timeseries.TimeSeriesDecompose", "decompose", f"period>len raised unexpected: {type(e).__name__}: {e}", "high")

# Edge 17: TimeSeriesDecompose with period=1 (no seasonality)
try:
    ts_p1 = TimeSeriesDecompose(np.arange(20, dtype=float), period=1, model="additive")
    result_p1 = ts_p1.decompose()
    print(f"  TimeSeriesDecompose period=1: seasonal range=[{result_p1.seasonal.min():.2f}, {result_p1.seasonal.max():.2f}]")
    if any(np.isnan(result_p1.seasonal)):
        report("timeseries.TimeSeriesDecompose", "decompose", "NaN in seasonal when period=1", "medium")
    else:
        print("  TimeSeriesDecompose period=1: OK")
except Exception as e:
    report("timeseries.TimeSeriesDecompose", "decompose", f"period=1 raised: {type(e).__name__}: {e}", "medium")

# Edge 18: MultiRegression with NaN in data
try:
    X_nan = np.array([[1, 2], [3, np.nan], [5, 6], [7, 8], [9, 10]], dtype=float)
    y_nan = np.array([1, 2, 3, 4, 5], dtype=float)
    mr_nan = MultiRegression(X_nan, y_nan)
    mr_nan.fit()
    r = mr_nan.result
    print(f"  MultiRegression NaN data: R2={r.r2}")
    if np.isnan(r.r2):
        report("stats.MultiRegression", "fit", "NaN R2 with NaN input data (no validation/warning)", "medium")
    else:
        print("  MultiRegression NaN: OK")
except Exception as e:
    print(f"  MultiRegression NaN data: raised {type(e).__name__} (acceptable)")

# Edge 19: EntropyWeight with all-identical columns
try:
    ident_data = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=float)
    ew_ident = EntropyWeight(ident_data)
    ew_ident.fit()
    w = ew_ident.weights
    print(f"  EntropyWeight identical columns: weights={w}")
    if any(np.isnan(w)):
        report("evaluate.EntropyWeight", "fit", "NaN weights with all-identical data", "high")
    elif not all(abs(wi - 1/3) < 0.01 for wi in w):
        report("evaluate.EntropyWeight", "fit", f"Identical columns should yield equal weights, got {w}", "medium")
    else:
        print("  EntropyWeight identical columns: OK")
except Exception as e:
    report("evaluate.EntropyWeight", "fit", f"Identical columns raised: {type(e).__name__}: {e}", "medium")

# Edge 20: DistributionFitter.fit with lognorm on negative data
try:
    neg_sample = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    df_neg = DistributionFitter(neg_sample)
    try:
        r_lognorm = df_neg.fit("lognorm")
        print(f"  DistributionFitter lognorm on negative: params={r_lognorm.params}")
    except Exception as e:
        print(f"  DistributionFitter lognorm on negative: raised {type(e).__name__} (expected)")
    try:
        r_expon = df_neg.fit("expon")
        print(f"  DistributionFitter expon on negative: params={r_expon.params}")
    except Exception as e:
        print(f"  DistributionFitter expon on negative: raised {type(e).__name__}")
    print("  DistributionFitter negative data: tested")
except Exception as e:
    report("prob.DistributionFitter", "fit", f"Negative data raised: {type(e).__name__}: {e}", "medium")

# Edge 21: TOPSIS with cost type only
try:
    cost_data = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    topsis_cost = TOPSIS(cost_data, types=[-1, -1])
    topsis_cost.fit()
    scores = topsis_cost.scores
    print(f"  TOPSIS all-cost: scores={scores}")
    # With all cost types, lowest values should be best
    assert scores[0] > scores[2], "With all-cost types, first row (lowest values) should have highest score"
    print("  TOPSIS all-cost: OK")
except Exception as e:
    report("evaluate.TOPSIS", "fit", f"All-cost types raised: {type(e).__name__}: {e}", "medium")

# Edge 22: FullPaper with special LaTeX characters
try:
    from mathflow.paper import FullPaper
    paper_special = FullPaper("Test $\\alpha$ & $\\beta$ # special chars %")
    paper_special.add_sub_problem("Test with ~ and ^", "Method")
    latex_special = paper_special.generate()
    print(f"  FullPaper special chars: generated {len(latex_special)} chars")
    print("  FullPaper special chars: OK")
except Exception as e:
    report("paper.FullPaper", "generate", f"Special LaTeX chars raised: {type(e).__name__}: {e}", "low")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"  TEST COMPLETE: {issue_counter[0]} issues found")
print("=" * 70)

if issues:
    print("\n  ISSUE SUMMARY:")
    for tag, module, method, error, severity in issues:
        print(f"  {tag}: [{module}.{method}] | ERROR: {error} | SEVERITY: {severity}")
else:
    print("  No issues found!")

if __name__ == "__main__":
    sys.exit(0 if not issues else 1)
