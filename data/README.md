# Data directory

Place raw data here. Expected schema (Zomato-style):

## Expected files / tables

- **users.csv**: user_id, segment (budget|premium|occasional), preferred_cuisines (or separate table), preferred_zones
- **restaurants.csv**: restaurant_id, cuisine, price_tier, rating_avg, delivery_rating, is_chain, city, zone
- **menu_items.csv**: item_id, restaurant_id, name, category (main|beverage|dessert|side|addon), veg (bool), price
- **orders.csv**: order_id, user_id, restaurant_id, created_at, total_value, city, zone
- **order_items.csv**: order_id, item_id, quantity, unit_price (for cart composition and add-on labels)
- **interactions.csv** (optional): user_id, item_id, restaurant_id, event_type (view|add|order), timestamp

## Sample data

Run `python -m src.data.generate_sample_data` to generate synthetic sample data for development.
