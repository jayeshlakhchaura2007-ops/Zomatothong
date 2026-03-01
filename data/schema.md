# CSAO Data Schema

Canonical entities and columns for feature pipeline and training.

## users
| Column        | Type   | Description                          |
|---------------|--------|--------------------------------------|
| user_id       | string | Unique user identifier               |
| segment       | string | budget, premium, occasional          |
| first_order_at| datetime | For recency/frequency               |

Derived (from orders): order_frequency, recency_days, monetary_value_avg, preferred_cuisines, preferred_zones.

## restaurants
| Column          | Type    | Description                |
|-----------------|---------|----------------------------|
| restaurant_id   | string  | Unique restaurant id       |
| cuisine         | string  | e.g. North Indian, Chinese |
| price_tier      | string  | low, mid, high             |
| rating_avg       | float   | Aggregate rating           |
| delivery_rating  | float   | Delivery score             |
| is_chain        | bool    | Chain vs independent       |
| city            | string  | City code or name          |
| zone            | string  | Delivery zone              |

## menu_items
| Column        | Type   | Description                    |
|---------------|--------|--------------------------------|
| item_id       | string | Unique item id                 |
| restaurant_id | string | Owning restaurant              |
| name          | string | Display name                   |
| category      | string | main, beverage, dessert, side, addon |
| veg           | bool   | Vegetarian flag                |
| price         | float  | Unit price                     |

## orders
| Column        | Type   | Description        |
|---------------|--------|--------------------|
| order_id      | string | Unique order id    |
| user_id       | string | User               |
| restaurant_id | string | Restaurant         |
| created_at    | datetime | Order timestamp  |
| total_value   | float  | Order total        |
| city          | string | Delivery city      |
| zone          | string | Delivery zone      |

## order_items
| Column    | Type  | Description   |
|-----------|-------|---------------|
| order_id  | string| Order         |
| item_id   | string| Menu item     |
| quantity  | int   | Quantity      |
| unit_price| float | Price at order|

Add-on label: for each order, items added after the first (or after a threshold) can be considered add-ons for training.
