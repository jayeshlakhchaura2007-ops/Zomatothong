"""
Generate synthetic sample data for CSAO pipeline development.
Schema matches data/README.md and data/schema.md.
"""
from pathlib import Path
import random
from datetime import datetime, timedelta

import pandas as pd

SEED = 42
random.seed(SEED)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
N_USERS = 200
N_RESTAURANTS = 30
N_ITEMS_PER_REST = 25
N_ORDERS = 1500
CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Beverages", "Bakery"]
SEGMENTS = ["budget", "premium", "occasional"]
PRICE_TIERS = ["low", "mid", "high"]
CATEGORIES = ["main", "beverage", "dessert", "side", "addon"]
CITIES = ["city_a", "city_b"]
ZONES = ["zone_1", "zone_2", "zone_3"]


def generate_users() -> pd.DataFrame:
    rows = []
    for i in range(N_USERS):
        rows.append({
            "user_id": f"u_{i}",
            "segment": random.choice(SEGMENTS),
            "first_order_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
        })
    return pd.DataFrame(rows)


def generate_restaurants() -> pd.DataFrame:
    rows = []
    for i in range(N_RESTAURANTS):
        rows.append({
            "restaurant_id": f"r_{i}",
            "cuisine": random.choice(CUISINES),
            "price_tier": random.choice(PRICE_TIERS),
            "rating_avg": round(random.uniform(3.5, 4.8), 1),
            "delivery_rating": round(random.uniform(3.5, 4.8), 1),
            "is_chain": random.random() < 0.3,
            "city": random.choice(CITIES),
            "zone": random.choice(ZONES),
        })
    return pd.DataFrame(rows)


def generate_menu_items(restaurants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    item_id = 0
    for _, r in restaurants.iterrows():
        rid = r["restaurant_id"]
        n = N_ITEMS_PER_REST
        for _ in range(n):
            cat = random.choices(CATEGORIES, weights=[4, 2, 2, 2, 2])[0]
            base_price = {"low": (80, 200), "mid": (150, 400), "high": (300, 800)}[r["price_tier"]]
            price = round(random.uniform(*base_price), 2)
            rows.append({
                "item_id": f"i_{item_id}",
                "restaurant_id": rid,
                "name": f"Item_{item_id}_{cat}",
                "category": cat,
                "veg": random.random() < 0.6,
                "price": price,
            })
            item_id += 1
    return pd.DataFrame(rows)


def generate_orders_and_items(
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    menu_items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    menu_by_rest = menu_items.groupby("restaurant_id")
    orders_list = []
    order_items_list = []
    base_time = datetime.now() - timedelta(days=180)

    for oid in range(N_ORDERS):
        uid = users.sample(1).iloc[0]["user_id"]
        rid = restaurants.sample(1).iloc[0]["restaurant_id"]
        rest = restaurants[restaurants["restaurant_id"] == rid].iloc[0]
        created = base_time + timedelta(seconds=random.randint(0, 180 * 86400))
        city = rest["city"]
        zone = rest["zone"]

        items_df = menu_by_rest.get_group(rid)
        n_items = random.randint(1, 6)
        chosen = items_df.sample(min(n_items, len(items_df)))
        total = (chosen["price"] * chosen.apply(lambda _: random.randint(1, 2), axis=1)).sum()
        total = round(float(total), 2)

        orders_list.append({
            "order_id": f"o_{oid}",
            "user_id": uid,
            "restaurant_id": rid,
            "created_at": created.isoformat(),
            "total_value": total,
            "city": city,
            "zone": zone,
        })
        for _, row in chosen.iterrows():
            qty = random.randint(1, 2)
            order_items_list.append({
                "order_id": f"o_{oid}",
                "item_id": row["item_id"],
                "quantity": qty,
                "unit_price": row["price"],
            })

    return pd.DataFrame(orders_list), pd.DataFrame(order_items_list)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users = generate_users()
    restaurants = generate_restaurants()
    menu_items = generate_menu_items(restaurants)
    orders, order_items = generate_orders_and_items(users, restaurants, menu_items)

    users.to_csv(DATA_DIR / "users.csv", index=False)
    restaurants.to_csv(DATA_DIR / "restaurants.csv", index=False)
    menu_items.to_csv(DATA_DIR / "menu_items.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    order_items.to_csv(DATA_DIR / "order_items.csv", index=False)
    print(f"Generated: users={len(users)}, restaurants={len(restaurants)}, menu_items={len(menu_items)}, orders={len(orders)}, order_items={len(order_items)}")
    print(f"Saved under {DATA_DIR}")


if __name__ == "__main__":
    main()
