"""Modeling - regression: used car prices (buy or pass at auction).
Run:  python regression.py   (expects ../data/used_car_sales.csv and ../data/auction_lots.csv)
Every number on the slides comes from this script.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
PREP = 1500  # cost to get a car ready for the lot
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)

cars = pd.read_csv(os.path.join(HERE, "used_car_sales.csv"))
lots = pd.read_csv(os.path.join(HERE, "auction_lots.csv"))
for d in (cars, lots):
    d["Accident"] = (d["Accident"] == "yes").astype(int)
print(f"cars rows={len(cars)}, auction lots={len(lots)}, mean sale price=${cars.Sale_Price.mean():,.0f}")

ALL = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
train, test = train_test_split(cars, test_size=0.30, random_state=SEED)
print(f"train={len(train)}, test={len(test)}")

def fit(features):
    X = sm.add_constant(train[features])
    return sm.OLS(train["Sale_Price"], X).fit()

# v1: all five features
m_all = fit(ALL)
print("\n=== FULL MODEL (all 5 features), training cars ===")
print(m_all.summary().tables[1])
print(f"R2={m_all.rsquared:.4f}  adj R2={m_all.rsquared_adj:.4f}")

# drop features that don't earn their place (p > 0.05), refit, compare adjusted R2
weak = [f for f in ALL if m_all.pvalues[f] > 0.05]
print("\nfeatures with p > 0.05:", weak or "none")
if weak:
    keep = [f for f in ALL if f not in weak]
    m_red = fit(keep)
    print(f"=== REDUCED MODEL (drop {weak}) ===")
    print(m_red.summary().tables[1])
    print(f"R2={m_red.rsquared:.4f}  adj R2={m_red.rsquared_adj:.4f}  (full adj R2={m_all.rsquared_adj:.4f})")
    use_reduced = m_red.rsquared_adj >= m_all.rsquared_adj
else:
    keep, m_red, use_reduced = ALL, m_all, False
v1_features = keep if use_reduced else ALL
v1 = m_red if use_reduced else m_all
print(f"\nv1 = {'reduced' if use_reduced else 'full'} model with features {v1_features}; training adj R2 = {v1.rsquared_adj:.4f}")

# 3. Test-set fit: baseline (training mean) vs v1
def predict(model, feats, df):
    return model.predict(sm.add_constant(df[feats], has_constant="add"))

base_mean = train["Sale_Price"].mean()
y = test["Sale_Price"].values
p_base = np.full(len(test), base_mean)
p_v1 = predict(v1, v1_features, test).values
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def rmse(y, p): return float(np.sqrt(((y - p) ** 2).mean()))
print(f"\nTEST CARS (n={len(test)})")
print(f"  baseline (train mean ${base_mean:,.0f}): R2={r2(y, p_base):.3f}  RMSE=${rmse(y, p_base):,.0f}")
print(f"  v1:                                  R2={r2(y, p_v1):.3f}  RMSE=${rmse(y, p_v1):,.0f}")

# 4. Decision rules applied to the 40 auction lots
def buy_rule(car, auction_price, model=v1, feats=v1_features):
    """car: dict of the five feature columns (Accident as 1/0). Returns 'buy' or 'pass'."""
    x = sm.add_constant(pd.DataFrame([car])[feats], has_constant="add")
    pred = float(model.predict(x).iloc[0])
    return "buy" if auction_price + PREP < pred else "pass"

def baseline_rule(auction_price):
    return "buy" if auction_price + PREP < base_mean else "pass"

lots["profit"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP
lots["pred"] = predict(v1, v1_features, lots).values
lots["v1_buy"] = [buy_rule(r[ALL].to_dict(), r["Auction_Price"]) == "buy" for _, r in lots.iterrows()]
lots["base_buy"] = [baseline_rule(p) == "buy" for p in lots["Auction_Price"]]
print(f"\nAUCTION LOTS (40 cars), profit = Later_Sale_Price - Auction_Price - ${PREP}")
for name, col in [("baseline", "base_buy"), ("v1", "v1_buy")]:
    b = lots[lots[col]]
    print(f"  {name:9s} buys {len(b):2d} cars, {int((b.profit < 0).sum()):2d} lose money, total profit ${b.profit.sum():,.0f}")
oracle = lots[lots.profit > 0]
print(f"  (hindsight best: buying every profitable car = {len(oracle)} cars, ${oracle.profit.sum():,.0f})")

# Example car through the function
ex = dict(Mileage_K=50, Age_Years=4, Horsepower=200, Accident=0, Num_Owners=2)
print(f"\nExample car {ex}: predicted sale price ${float(v1.predict(sm.add_constant(pd.DataFrame([ex])[v1_features], has_constant='add')).iloc[0]):,.0f}")

# Chart: predicted vs actual on test cars
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y, p_v1, s=22, label=f"v1 (R²={r2(y, p_v1):.2f})", color="#1f4e9c")
ax.scatter(y, p_base, s=14, marker="_", label=f"baseline (R²={r2(y, p_base):.2f})", color="#999999")
lim = [min(y.min(), p_v1.min()) - 1000, max(y.max(), p_v1.max()) + 1000]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("Actual sale price ($)"); ax.set_ylabel("Predicted sale price ($)")
ax.set_title("Test cars: predicted vs actual"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pred_vs_actual.png"), dpi=200)
print("saved out/pred_vs_actual.png")
print("\nv1 coefficients:\n" + v1.params.round(2).to_string())
