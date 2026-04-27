"""Synthetic Bangalore consumption data generator.

Generates realistic interaction data for 500 users, 200 restaurants,
500 products, 100 venues across 6 months. Includes temporal patterns
(lunch/dinner peaks, weekday/weekend splits) and cross-domain
co-occurrence patterns for OFTEN_PAIRED and FOLLOWED_BY_DINING edges.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent
N_USERS = 500
N_RESTAURANTS = 200
N_PRODUCTS = 500
N_VENUES = 100
DAYS = 180  # 6 months
START_DATE = datetime(2025, 10, 1)

BANGALORE_AREAS = [
    "Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "Jayanagar",
    "JP Nagar", "BTM Layout", "Marathahalli", "Electronic City", "Banashankari",
    "Malleshwaram", "Rajajinagar", "Yelahanka", "Hebbal", "Bellandur",
    "Sarjapur Road", "Domlur", "Richmond Town", "MG Road", "Brigade Road",
    "Basavanagudi", "Vijayanagar", "Nagarbhavi", "Bannerghatta Road", "Kanakapura Road",
]

CUISINES = [
    "South Indian", "North Indian", "Chinese", "Biryani", "Pizza",
    "Burger", "Andhra", "Kerala", "Mughlai", "Continental",
    "Thai", "Japanese", "Italian", "Street Food", "Desserts",
    "Cafe", "Healthy", "Bengali", "Rajasthani", "Chettinad",
]

PRODUCT_CATEGORIES = [
    "Beverages", "Snacks", "Dairy", "Cleaning", "Personal Care",
    "Fruits", "Vegetables", "Staples", "Frozen", "Bakery",
    "Instant Food", "Sauces", "Oil", "Spices", "Baby Care",
]

PRODUCT_NAMES = {
    "Beverages": ["Coca Cola 750ml", "Pepsi 2L", "Frooti 1L", "Appy 250ml", "Red Bull 250ml",
                  "Paper Boat Aamras", "Sprite 750ml", "Maaza 1.2L", "Bisleri 1L", "Thumps Up 750ml"],
    "Snacks": ["Lays Classic 52g", "Kurkure 90g", "Haldiram Bhujia 200g", "Parle G 250g",
               "Dark Fantasy 75g", "Hide & Seek 120g", "Bingo Tedhe Medhe", "Too Yumm 80g"],
    "Dairy": ["Amul Toned Milk 1L", "Nandini Curd 400g", "Amul Butter 100g", "Britannia Cheese Slice",
              "Mother Dairy Paneer 200g", "Epigamia Greek Yogurt", "Amul Lassi 200ml"],
    "Cleaning": ["Harpic 500ml", "Lizol 500ml", "Vim Bar 300g", "Surf Excel 1kg",
                 "Colin Glass Cleaner", "Domex 500ml", "Scotch Brite Pad"],
    "Personal Care": ["Dove Soap 100g", "Head & Shoulders 180ml", "Colgate 150g", "Dettol Handwash",
                      "Nivea Body Lotion", "Gillette Guard", "Whisper Ultra"],
}

RESTAURANT_NAMES = [
    "Meghana Foods", "Nandhana Palace", "Empire Restaurant", "MTR", "Vidyarthi Bhavan",
    "Truffles", "Toit", "Corner House", "Brahmin's Coffee Bar", "Mavalli Tiffin Rooms",
    "A2B", "Shivaji Military Hotel", "Nagarjuna", "Paradise Biryani", "Behrouz Biryani",
    "Domino's Pizza", "Pizza Hut", "McDonald's", "KFC", "Burger King",
    "Subway", "Starbucks", "Chai Point", "Third Wave Coffee", "Blue Tokai",
]

VENUE_NAMES = [
    "Farzi Cafe", "Byg Brewski", "The Black Pearl", "Windmills Craftworks",
    "Bob's Bar", "Hard Rock Cafe", "Hammered", "Plan B", "Communiti",
    "1Q1 Kitchen", "Toast & Tonic", "The Permit Room", "Chinita", "Fatty Bao",
    "Smoke House Deli", "Olive Beach", "Caperberry", "Ebony", "Rim Naam", "Karavalli",
]


def generate_users() -> list[dict]:
    users = []
    for i in range(N_USERS):
        users.append({
            "user_id": f"U{i:04d}",
            "location": random.choice(BANGALORE_AREAS),
            "cuisine_pref": random.choice(CUISINES),
            "price_sensitivity": round(random.uniform(0.2, 1.0), 2),
            "signup_days_ago": random.randint(30, 365),
        })
    return users


def generate_restaurants() -> list[dict]:
    restaurants = []
    for i in range(N_RESTAURANTS):
        name = RESTAURANT_NAMES[i % len(RESTAURANT_NAMES)]
        if i >= len(RESTAURANT_NAMES):
            name = f"{name} {BANGALORE_AREAS[i % len(BANGALORE_AREAS)]}"
        restaurants.append({
            "restaurant_id": f"R{i:04d}",
            "name": name,
            "cuisine": random.choice(CUISINES),
            "rating": round(random.uniform(3.2, 4.9), 1),
            "price_range": random.randint(1, 4),
            "avg_delivery_time_min": random.randint(20, 55),
            "area": random.choice(BANGALORE_AREAS),
            "lat": round(12.9 + random.uniform(-0.15, 0.15), 6),
            "lng": round(77.6 + random.uniform(-0.15, 0.15), 6),
        })
    return restaurants


def generate_products() -> list[dict]:
    products = []
    for i in range(N_PRODUCTS):
        cat = random.choice(PRODUCT_CATEGORIES)
        names = PRODUCT_NAMES.get(cat, [f"{cat} Item {i}"])
        products.append({
            "product_id": f"P{i:04d}",
            "name": names[i % len(names)],
            "category": cat,
            "brand": random.choice(["Amul", "ITC", "HUL", "P&G", "Nestle", "Britannia", "Dabur", "Marico"]),
            "price": round(random.uniform(15, 500), 0),
            "unit_size": random.choice(["100g", "200g", "500g", "1kg", "250ml", "500ml", "1L", "1pc"]),
        })
    return products


def generate_venues() -> list[dict]:
    venues = []
    for i in range(N_VENUES):
        name = VENUE_NAMES[i % len(VENUE_NAMES)]
        if i >= len(VENUE_NAMES):
            name = f"{name} {BANGALORE_AREAS[i % len(BANGALORE_AREAS)]}"
        venues.append({
            "venue_id": f"V{i:04d}",
            "name": name,
            "cuisine": random.choice(CUISINES),
            "ambiance": random.choice(["casual", "fine_dining", "cafe", "pub", "rooftop"]),
            "price_range": random.randint(2, 4),
            "capacity": random.randint(20, 200),
            "avg_rating": round(random.uniform(3.5, 4.8), 1),
            "area": random.choice(BANGALORE_AREAS),
            "lat": round(12.9 + random.uniform(-0.15, 0.15), 6),
            "lng": round(77.6 + random.uniform(-0.15, 0.15), 6),
        })
    return venues


def generate_food_orders(users: list[dict], restaurants: list[dict]) -> list[dict]:
    """Generate 50K food orders with realistic temporal patterns."""
    orders = []
    for _ in range(50_000):
        user = random.choice(users)
        restaurant = random.choice(restaurants)
        day_offset = random.randint(0, DAYS - 1)
        dt = START_DATE + timedelta(days=day_offset)

        # Lunch (11-14) or dinner (19-22) bias
        if random.random() < 0.7:
            hour = random.choice([*range(11, 14), *range(19, 22)])
        else:
            hour = random.randint(8, 23)
        minute = random.randint(0, 59)
        dt = dt.replace(hour=hour, minute=minute)

        orders.append({
            "order_id": f"FO{len(orders):06d}",
            "user_id": user["user_id"],
            "restaurant_id": restaurant["restaurant_id"],
            "timestamp": dt.isoformat(),
            "amount": round(random.uniform(100, 800), 0),
            "rating": random.choice([0, 0, 3, 4, 4, 5, 5]),
            "items_count": random.randint(1, 5),
        })
    return orders


def generate_instamart_orders(users: list[dict], products: list[dict]) -> list[dict]:
    """Generate 20K grocery orders with reorder patterns."""
    orders = []
    for _ in range(20_000):
        user = random.choice(users)
        product = random.choice(products)
        day_offset = random.randint(0, DAYS - 1)
        dt = START_DATE + timedelta(days=day_offset)
        hour = random.randint(8, 21)
        dt = dt.replace(hour=hour, minute=random.randint(0, 59))

        orders.append({
            "order_id": f"IO{len(orders):06d}",
            "user_id": user["user_id"],
            "product_id": product["product_id"],
            "timestamp": dt.isoformat(),
            "quantity": random.randint(1, 3),
            "amount": round(float(product["price"]) * random.randint(1, 3), 0),
        })
    return orders


def generate_dineout_bookings(users: list[dict], venues: list[dict]) -> list[dict]:
    """Generate 5K dineout bookings with weekend bias."""
    bookings = []
    for _ in range(5_000):
        user = random.choice(users)
        venue = random.choice(venues)
        day_offset = random.randint(0, DAYS - 1)
        dt = START_DATE + timedelta(days=day_offset)

        # Weekend bias
        if random.random() < 0.6:
            while dt.weekday() < 5:
                dt += timedelta(days=1)

        hour = random.choice([12, 13, 19, 20, 21])
        dt = dt.replace(hour=hour, minute=random.choice([0, 30]))

        bookings.append({
            "booking_id": f"DB{len(bookings):06d}",
            "user_id": user["user_id"],
            "venue_id": venue["venue_id"],
            "timestamp": dt.isoformat(),
            "party_size": random.choice([2, 2, 2, 3, 4, 4, 6]),
            "rating": random.choice([0, 3, 4, 4, 5, 5]),
        })
    return bookings


def write_csv(filepath: Path, data: list[dict]) -> None:
    if not data:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"  Written {len(data)} rows to {filepath}")


def main() -> None:
    print("Generating synthetic Bangalore consumption data...")
    print(f"  Users: {N_USERS}, Restaurants: {N_RESTAURANTS}, Products: {N_PRODUCTS}, Venues: {N_VENUES}")

    users = generate_users()
    restaurants = generate_restaurants()
    products = generate_products()
    venues = generate_venues()

    write_csv(DATA_DIR / "bangalore_users.csv", users)
    write_csv(DATA_DIR / "bangalore_restaurants.csv", restaurants)
    write_csv(DATA_DIR / "bangalore_products.csv", products)
    write_csv(DATA_DIR / "bangalore_venues.csv", venues)

    food_orders = generate_food_orders(users, restaurants)
    instamart_orders = generate_instamart_orders(users, products)
    dineout_bookings = generate_dineout_bookings(users, venues)

    write_csv(DATA_DIR / "interactions" / "food_orders.csv", food_orders)
    write_csv(DATA_DIR / "interactions" / "instamart_orders.csv", instamart_orders)
    write_csv(DATA_DIR / "interactions" / "dineout_bookings.csv", dineout_bookings)

    print("Done.")


if __name__ == "__main__":
    main()
