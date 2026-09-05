"""
Build realistic multi-order transaction log for Instacart grocery shoppers.
Merges historical prior orders and final evaluation order for a cohort of 1,000 shoppers.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "instacart"
out_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "d2_advanced" / "instacart_transactions.csv"

print("Step 1: Reading orders.csv...")
orders = pd.read_csv(raw_dir / "orders.csv")
train_orders = orders[orders["eval_set"] == "train"]
sample_users = train_orders["user_id"].unique()[:1000]

user_orders = orders[orders["user_id"].isin(sample_users)].copy()
prior_order_ids = set(user_orders.loc[user_orders["eval_set"] == "prior", "order_id"])
train_order_ids = set(user_orders.loc[user_orders["eval_set"] == "train", "order_id"])
print(f"Target users: {len(sample_users)}, Prior orders: {len(prior_order_ids)}, Train orders: {len(train_order_ids)}")

print("Step 2: Loading products and taxonomy...")
products = pd.read_csv(raw_dir / "products.csv")
depts = pd.read_csv(raw_dir / "departments.csv")
aisles = pd.read_csv(raw_dir / "aisles.csv")
prod_info = products.merge(depts, on="department_id", how="left").merge(aisles, on="aisle_id", how="left")
prod_map = prod_info.set_index("product_id")[["product_name", "department", "aisle"]].to_dict(orient="index")

print("Step 3: Loading train order line items...")
train_items = pd.read_csv(raw_dir / "order_products__train.csv")
train_items = train_items[train_items["order_id"].isin(train_order_ids)].copy()

print("Step 4: Streaming and filtering prior order line items...")
t0 = time.time()
prior_chunks = []
chunk_size = 1_000_000
for chunk in pd.read_csv(raw_dir / "order_products__prior.csv", chunksize=chunk_size):
    matched = chunk[chunk["order_id"].isin(prior_order_ids)]
    if not matched.empty:
        prior_chunks.append(matched)

prior_items = pd.concat(prior_chunks, ignore_index=True)
print(f"Filtered {len(prior_items):,} prior items and {len(train_items):,} train items in {time.time()-t0:.1f}s.")

all_items = pd.concat([prior_items, train_items], ignore_index=True)
print(f"Total basket lines: {len(all_items):,}")

print("Step 5: Merging order metadata and product details...")
merged = all_items.merge(
    user_orders[["order_id", "user_id", "order_number", "days_since_prior_order"]],
    on="order_id",
    how="left",
)

# Harmonize column names to standard schema
merged = merged.rename(
    columns={
        "order_id": "InvoiceNo",
        "user_id": "CustomerID",
        "product_id": "StockCode",
        "order_number": "OrderSequence",
    }
)
merged["Quantity"] = 1
merged["DaysSincePriorOrder"] = merged["days_since_prior_order"].fillna(0)

# Map descriptions and categories
names = []
cats = []
subcats = []
for pid in merged["StockCode"]:
    info = prod_map.get(pid, {})
    names.append(info.get("product_name", "Grocery Item"))
    cats.append(info.get("department", "grocery"))
    subcats.append(info.get("aisle", "staple"))

merged["Description"] = names
merged["Category"] = cats
merged["SubCategory"] = subcats

# Save to destination
out_csv.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_csv, index=False)
print(f"Successfully generated {out_csv} with {len(merged):,} rows across {merged['CustomerID'].nunique()} shoppers!")
