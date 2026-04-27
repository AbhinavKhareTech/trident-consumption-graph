"""Shared test fixtures for BGI Trident."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_users():
    return pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(10)],
        "location": ["Koramangala"] * 5 + ["Indiranagar"] * 5,
        "cuisine_pref": ["Biryani", "South Indian"] * 5,
        "price_sensitivity": np.random.uniform(0.3, 0.9, 10).round(2),
        "signup_days_ago": np.random.randint(30, 200, 10),
    })

@pytest.fixture
def sample_restaurants():
    return pd.DataFrame({
        "restaurant_id": [f"R{i:04d}" for i in range(5)],
        "name": ["Meghana's", "Nandhana", "Truffles", "MTR", "Empire"],
        "cuisine": ["Biryani", "Andhra", "Continental", "South Indian", "Biryani"],
        "rating": [4.5, 4.3, 4.4, 4.6, 4.1],
        "price_range": [2, 2, 3, 2, 2],
        "avg_delivery_time_min": [35, 30, 40, 25, 30],
        "area": ["Koramangala", "Koramangala", "Indiranagar", "Jayanagar", "Brigade Road"],
        "lat": [12.93, 12.93, 12.97, 12.92, 12.97],
        "lng": [77.62, 77.63, 77.64, 77.58, 77.60],
    })

@pytest.fixture
def sample_products():
    return pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(5)],
        "name": ["Coke 750ml", "Harpic 500ml", "Amul Milk 1L", "Lays 52g", "Surf Excel 1kg"],
        "category": ["Beverages", "Cleaning", "Dairy", "Snacks", "Cleaning"],
        "brand": ["Coca Cola", "Reckitt", "Amul", "PepsiCo", "HUL"],
        "price": [40, 189, 60, 20, 245],
        "unit_size": ["750ml", "500ml", "1L", "52g", "1kg"],
    })

@pytest.fixture
def sample_venues():
    return pd.DataFrame({
        "venue_id": [f"V{i:04d}" for i in range(3)],
        "name": ["Farzi Cafe", "Toit", "Karavalli"],
        "cuisine": ["North Indian", "Continental", "Kerala"],
        "ambiance": ["fine_dining", "pub", "fine_dining"],
        "price_range": [4, 3, 4],
        "capacity": [100, 150, 80],
        "avg_rating": [4.3, 4.4, 4.6],
        "area": ["UB City", "Indiranagar", "Residency Road"],
        "lat": [12.97, 12.98, 12.96],
        "lng": [77.60, 77.64, 77.59],
    })

@pytest.fixture
def sample_food_orders():
    base = datetime(2025, 10, 1)
    rows = []
    for i in range(100):
        rows.append({
            "order_id": f"FO{i:06d}",
            "user_id": f"U{i % 10:04d}",
            "restaurant_id": f"R{i % 5:04d}",
            "timestamp": (base + timedelta(days=i % 60, hours=12 + i % 10)).isoformat(),
            "amount": 200 + (i % 5) * 100,
            "rating": [4, 5, 3, 4, 5][i % 5],
            "items_count": 1 + i % 3,
        })
    return pd.DataFrame(rows)

@pytest.fixture
def sample_instamart_orders():
    base = datetime(2025, 10, 1)
    rows = []
    for i in range(50):
        rows.append({
            "order_id": f"IO{i:06d}",
            "user_id": f"U{i % 10:04d}",
            "product_id": f"P{i % 5:04d}",
            "timestamp": (base + timedelta(days=i % 60, hours=10 + i % 8)).isoformat(),
            "quantity": 1 + i % 2,
            "amount": [40, 189, 60, 20, 245][i % 5],
        })
    return pd.DataFrame(rows)

@pytest.fixture
def sample_dineout_bookings():
    base = datetime(2025, 10, 1)
    rows = []
    for i in range(20):
        rows.append({
            "booking_id": f"DB{i:06d}",
            "user_id": f"U{i % 10:04d}",
            "venue_id": f"V{i % 3:04d}",
            "timestamp": (base + timedelta(days=i * 7 + 5, hours=19 + i % 3)).isoformat(),
            "party_size": [2, 4, 2, 3, 6][i % 5],
            "rating": [4, 5, 4, 5, 3][i % 5],
        })
    return pd.DataFrame(rows)
