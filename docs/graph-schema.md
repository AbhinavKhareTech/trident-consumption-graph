# Graph Schema Reference

See `src/bgi_trident/graph/schema.py` for the canonical definitions.

## Node Types (6)

| Type | Embedding Dim | Features |
|---|---|---|
| User | 128 | cuisine_pref_vector (20), price_sensitivity, order_frequency, avg_order_value, preferred_hour_vector (24), preferred_day_vector (7) |
| Restaurant | 128 | cuisine_vector (20), rating, price_range, avg_delivery_time_min, geohash_embedding (8), is_promoted |
| Product | 64 | category_vector (50), brand_embedding (16), unit_price, unit_size, avg_reorder_interval_days |
| Venue | 64 | cuisine_vector (20), ambiance_vector, price_range, avg_rating, capacity, geohash_embedding (8) |
| TimeSlot | 16 | hour_of_day, day_of_week, is_weekend, is_holiday, is_meal_hour |
| Location | 16 | geohash_6, area_name, lat, lng, restaurant_density |

## Edge Types (8)

| Edge | Source | Target | Weight Signals | Temporal Decay |
|---|---|---|---|---|
| ORDERED_FROM | User | Restaurant | order_count, recency_days, avg_spend_inr, avg_rating_given | Yes |
| PURCHASED | User | Product | purchase_count, avg_interval_days, last_purchase_days_ago | Yes |
| BOOKED_AT | User | Venue | booking_count, avg_party_size, avg_rating_given | Yes |
| PREFERS_AT_TIME | User | TimeSlot | order_count_at_slot, conversion_rate_at_slot | No |
| LOCATED_NEAR | User | Location | delivery_count_to_location, is_primary_address | No |
| SERVES_CUISINE | Restaurant | Restaurant | cuisine_overlap_score | No |
| OFTEN_PAIRED | Restaurant | Product | co_occurrence_count, co_occurrence_within_hours | Yes |
| FOLLOWED_BY_DINING | Restaurant | Venue | sequence_count, avg_gap_hours | Yes |

## Cross-Domain Edge Discovery

**OFTEN_PAIRED**: discovered when a food order and instamart purchase occur on the same day for the same user (minimum 2 co-occurrences).

**FOLLOWED_BY_DINING**: discovered when a food order is followed by a dineout booking within 48 hours (minimum 2 sequences).

These edges are what DoorDash's single-domain graph cannot represent.
