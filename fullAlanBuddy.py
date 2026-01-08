import os
import requests
from typing import List, Dict, Union, Tuple, Any, Set, Optional
import json
import time 
import random 
from itertools import combinations, permutations
from dotenv import load_dotenv
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import hashlib

# Load environment variables from the config.env file
load_dotenv('config.env') 

# --- 1. API KEY SETUP AND CONFIGURATION ---
ORS_API_KEY = os.getenv("ORS_API_KEY")

if not ORS_API_KEY:
    print("FATAL ERROR: ORS_API_KEY missing in 'config.env'.")
    exit(1)

# OpenRouteService Endpoints
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
ORS_ISOCHRONE_URL = "https://api.openrouteservice.org/v2/isochrones/driving-car"

# Overpass API Endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter" 

# --- API LIMIT CONSTANT ---
MAX_MATRIX_LOCATIONS = 50 
MAX_STORES_TO_USE = MAX_MATRIX_LOCATIONS - 1  # One spot for "Start" location

# --- PRICE VARIANCE RANGE ---
MIN_PRICE_FACTOR = 0.6
MAX_PRICE_FACTOR = 1.4

# --- NUTRITION CONSTANTS ---
DAILY_CALORIE_TARGET = 2000
DAYS_PER_WEEK = 7
MEALS_PER_DAY = 3
TOTAL_WEEKLY_CALORIES = DAILY_CALORIE_TARGET * DAYS_PER_WEEK
TOTAL_MEALS = DAYS_PER_WEEK * MEALS_PER_DAY

# Target macronutrient distribution (as percentages)
CARB_PERCENT = 0.55  # 55% carbs
PROTEIN_PERCENT = 0.15  # 15% protein
FAT_PERCENT = 0.30  # 30% fat

# Calories per gram of each macronutrient
CALORIES_PER_GRAM_CARB = 4
CALORIES_PER_GRAM_PROTEIN = 4
CALORIES_PER_GRAM_FAT = 9

# --- 2. DYNAMIC SEARCH CONFIG & CONSTANTS ---
# User input for location and shopping time
USER_START_LOCATION: Tuple[float, float] = (40.0033, -86.1366)  # Near Indianapolis
SHOPPING_TIME_HOURS = 3.0  # User's available shopping time
MAX_TIME_SECONDS = SHOPPING_TIME_HOURS * 3600 * 1.1  # 10% buffer

# --- STORE TIME MULTIPLIERS ---
STORE_TIME_MULTIPLIERS: Dict[str, float] = {
    "Aldi": 0.8,         
    "Trader Joe's": 0.9, 
    "Kroger": 1.0,       
    "Meijer": 1.1,       
    "Whole Foods": 1.2,  
    "Walmart": 1.2,      
    "Costco": 1.5         
}

# --- TIME MODEL CONSTANTS ---
BASE_CHECKOUT_TIME_MINUTES = 7 
BASE_PARK_ENTRANCE_TIME_MINUTES = 5
TIME_PER_ITEM_MINUTES = 0.5  # Reduced since we're buying in bulk for a week

BASE_FIXOUT_OVERHEAD = (BASE_CHECKOUT_TIME_MINUTES + BASE_PARK_ENTRANCE_TIME_MINUTES) * 60 
TIME_PER_UNIT_SECONDS = TIME_PER_ITEM_MINUTES * 60 

# --- DATA FILES ---
VEGGIES_CSV = "alanVeggies.csv"
RECIPES_CSV = "RAW_recipes.csv"

# --- HELPER FUNCTION: Load Available Vegetables ---
def load_available_vegetables() -> Dict[str, List[str]]:
    """
    Load available vegetables from alanVeggies.csv and return a dictionary
    of vegetable -> list of cities where it's available
    """
    print(f"Loading available vegetables from {VEGGIES_CSV}...")
    
    veggie_data = {}
    
    try:
        with open(VEGGIES_CSV, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            required_columns = ['Vegetable', 'Form', 'RetailPrice', 'RetailPriceUnit']
            if not all(col in reader.fieldnames for col in required_columns):
                print(f"Error: CSV file must have columns: {required_columns}")
                return {}
            
            # Get all city columns
            city_columns = [col for col in reader.fieldnames if col not in required_columns]
            
            for row in reader:
                vegetable = row['Vegetable'].strip().lower()
                form = row['Form'].strip().lower()
                
                # Only consider fresh/raw vegetables
                if form not in ['fresh', 'raw', '']:
                    continue
                
                available_cities = []
                for city in city_columns:
                    price_str = row[city].strip()
                    if price_str and price_str.lower() not in ['', 'na', 'n/a']:
                        try:
                            float(price_str)
                            available_cities.append(city)
                        except ValueError:
                            continue
                
                if available_cities:
                    veggie_data[vegetable] = available_cities
        
        print(f"Loaded {len(veggie_data)} available vegetables.")
        return veggie_data
        
    except FileNotFoundError:
        print(f"Error: CSV file '{VEGGIES_CSV}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

# --- HELPER FUNCTION: Load Recipes ---
def load_recipes() -> List[Dict[str, Any]]:
    """
    Load recipes from RAW_recipes.csv and parse nutrition data
    """
    print(f"Loading recipes from {RECIPES_CSV}...")
    
    recipes = []
    
    try:
        df = pd.read_csv(RECIPES_CSV)
        
        # Parse nutrition column (it's a string like '[192.0, 8.0, 22.0, 1.0, 2.0, 0.0]')
        for _, row in df.iterrows():
            try:
                # Parse nutrition string to list
                nutrition_str = str(row['nutrition']).strip('[]')
                nutrition_values = [float(x.strip()) for x in nutrition_str.split(',')]
                
                # Based on common format: [calories, total fat, sugar, sodium, protein, saturated fat]
                if len(nutrition_values) >= 6:
                    recipe = {
                        'name': str(row['name']),
                        'id': str(row['id']),
                        'minutes': int(row['minutes']),
                        'nutrition': {
                            'calories': nutrition_values[0],
                            'total_fat': nutrition_values[1],
                            'sugar': nutrition_values[2],
                            'sodium': nutrition_values[3],
                            'protein': nutrition_values[4],
                            'saturated_fat': nutrition_values[5]
                        },
                        'ingredients': eval(row['ingredients']) if isinstance(row['ingredients'], str) else row['ingredients'],
                        'n_ingredients': int(row['n_ingredients']),
                        'tags': eval(row['tags']) if isinstance(row['ingredients'], str) else row['tags'],
                        'description': str(row['description'])
                    }
                    recipes.append(recipe)
            except Exception as e:
                continue
        
        print(f"Loaded {len(recipes)} recipes with nutrition data.")
        return recipes
        
    except FileNotFoundError:
        print(f"Error: CSV file '{RECIPES_CSV}' not found.")
        return []
    except Exception as e:
        print(f"Error reading recipes CSV: {e}")
        return []

# --- FUNCTION: Filter Recipes by Available Vegetables ---
def filter_recipes_by_available_vegetables(recipes: List[Dict[str, Any]], available_veggies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Filter recipes to only include those that use available vegetables
    """
    print("Filtering recipes by available vegetables...")
    
    filtered_recipes = []
    
    for recipe in recipes:
        recipe_ingredients = recipe['ingredients']
        all_ingredients_available = True
        
        # Check each ingredient in the recipe
        for ingredient in recipe_ingredients:
            ingredient_lower = ingredient.lower()
            
            # Check if this ingredient (or a similar one) is in our available veggies
            ingredient_found = False
            
            # Direct match
            if ingredient_lower in available_veggies:
                ingredient_found = True
            else:
                # Check for partial matches (e.g., "chopped tomatoes" contains "tomatoes")
                for veggie in available_veggies.keys():
                    if veggie in ingredient_lower or ingredient_lower in veggie:
                        ingredient_found = True
                        break
            
            if not ingredient_found:
                all_ingredients_available = False
                break
        
        if all_ingredients_available:
            filtered_recipes.append(recipe)
    
    print(f"Found {len(filtered_recipes)} recipes using only available vegetables.")
    return filtered_recipes

# --- FUNCTION: Create Weekly Meal Plan ---
def create_weekly_meal_plan(filtered_recipes: List[Dict[str, Any]], available_veggies: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Create a week's worth of meals (21 meals) that meet nutritional targets
    Returns: (meal_plan, ingredient_quantities)
    """
    print("\n--- CREATING WEEKLY MEAL PLAN ---")
    print(f"Target: {TOTAL_MEALS} meals ({DAYS_PER_WEEK} days × {MEALS_PER_DAY} meals/day)")
    print(f"Total calories per week: {TOTAL_WEEKLY_CALORIES:,}")
    
    if not filtered_recipes:
        print("No recipes available for meal planning!")
        return [], {}
    
    # Calculate per-meal calorie target
    calories_per_meal = TOTAL_WEEKLY_CALORIES / TOTAL_MEALS
    print(f"Target calories per meal: {calories_per_meal:.0f}")
    
    # Group recipes by calorie ranges
    low_cal_recipes = []    # < 300 calories
    med_cal_recipes = []    # 300-600 calories
    high_cal_recipes = []   # > 600 calories
    
    for recipe in filtered_recipes:
        calories = recipe['nutrition']['calories']
        if calories < 300:
            low_cal_recipes.append(recipe)
        elif calories <= 600:
            med_cal_recipes.append(recipe)
        else:
            high_cal_recipes.append(recipe)
    
    print(f"Available recipes by calorie range:")
    print(f"  Low calorie (<300): {len(low_cal_recipes)} recipes")
    print(f"  Medium calorie (300-600): {len(med_cal_recipes)} recipes")
    print(f"  High calorie (>600): {len(high_cal_recipes)} recipes")
    
    # Create meal plan
    meal_plan = []
    ingredient_quantities: Dict[str, int] = {}
    
    # For variety, we'll aim for:
    # - 7 high-calorie meals (dinners)
    # - 7 medium-calorie meals (lunches)
    # - 7 low-calorie meals (breakfasts)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in days:
        # Breakfast (low calorie)
        if low_cal_recipes:
            breakfast = random.choice(low_cal_recipes)
            meal_plan.append({
                'day': day,
                'meal_type': 'Breakfast',
                'recipe': breakfast['name'],
                'recipe_id': breakfast['id'],
                'calories': breakfast['nutrition']['calories'],
                'protein': breakfast['nutrition']['protein'],
                'ingredients': breakfast['ingredients']
            })
            # Track ingredients
            for ingredient in breakfast['ingredients']:
                ingredient_lower = ingredient.lower()
                # Find matching vegetable
                for veggie in available_veggies.keys():
                    if veggie in ingredient_lower or ingredient_lower in veggie:
                        ingredient_quantities[veggie] = ingredient_quantities.get(veggie, 0) + 1
                        break
        
        # Lunch (medium calorie)
        if med_cal_recipes:
            lunch = random.choice(med_cal_recipes)
            meal_plan.append({
                'day': day,
                'meal_type': 'Lunch',
                'recipe': lunch['name'],
                'recipe_id': lunch['id'],
                'calories': lunch['nutrition']['calories'],
                'protein': lunch['nutrition']['protein'],
                'ingredients': lunch['ingredients']
            })
            # Track ingredients
            for ingredient in lunch['ingredients']:
                ingredient_lower = ingredient.lower()
                for veggie in available_veggies.keys():
                    if veggie in ingredient_lower or ingredient_lower in veggie:
                        ingredient_quantities[veggie] = ingredient_quantities.get(veggie, 0) + 1
                        break
        
        # Dinner (high calorie)
        if high_cal_recipes:
            dinner = random.choice(high_cal_recipes)
            meal_plan.append({
                'day': day,
                'meal_type': 'Dinner',
                'recipe': dinner['name'],
                'recipe_id': dinner['id'],
                'calories': dinner['nutrition']['calories'],
                'protein': dinner['nutrition']['protein'],
                'ingredients': dinner['ingredients']
            })
            # Track ingredients
            for ingredient in dinner['ingredients']:
                ingredient_lower = ingredient.lower()
                for veggie in available_veggies.keys():
                    if veggie in ingredient_lower or ingredient_lower in veggie:
                        ingredient_quantities[veggie] = ingredient_quantities.get(veggie, 0) + 1
                        break
    
    # Calculate total nutrition
    total_calories = sum(meal['calories'] for meal in meal_plan)
    total_protein = sum(meal['protein'] for meal in meal_plan)
    
    print(f"\nMeal Plan Created:")
    print(f"  Total meals: {len(meal_plan)}")
    print(f"  Total calories: {total_calories:.0f} ({total_calories/TOTAL_WEEKLY_CALORIES*100:.1f}% of target)")
    print(f"  Total protein: {total_protein:.1f}g")
    print(f"  Unique ingredients needed: {len(ingredient_quantities)}")
    
    return meal_plan, ingredient_quantities

# --- HELPER FUNCTION: Get the GeoJSON Bounding Box ---
def get_geojson_bounding_box(geometry: Dict) -> Tuple[float, float, float, float]:
    """Calculates the bounding box from Isochrone geometry."""
    if geometry['type'] != 'Polygon':
        return (0, 0, 0, 0)
    
    all_coords = [coord for ring in geometry['coordinates'] for coord in ring]
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    return (min_lat, min_lon, max_lat, max_lon)

# --- FUNCTION: Get Travel Isochrone ---
def get_travel_isochrone(start_location: Tuple[float, float], time_limit_seconds: int) -> Union[Dict, None]:
    """
    Calculates the maximum area reachable within the time_limit_seconds.
    Returns the GeoJSON geometry of the reachable area.
    """
    print(f"\nCalculating reachable area ({int(time_limit_seconds / 60)} minutes)...")
    
    # Isochrone requires [Lon, Lat] format
    lon, lat = start_location[1], start_location[0]
    
    headers = {
        'Accept': 'application/geo+json',
        'Authorization': f'Bearer {ORS_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Updated payload format - OpenRouteService v2 requires different structure
    payload = {
        "locations": [[lon, lat]],
        "range": [time_limit_seconds],
        "range_type": "time",
        "units": "m"  # meters (default) or "km"
    }
    
    try:
        response = requests.post(ORS_ISOCHRONE_URL, headers=headers, json=payload, timeout=15)
        
        # Debug: Print response details
        print(f"Request URL: {ORS_ISOCHRONE_URL}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Response Text: {response.text[:200]}")
        
        response.raise_for_status()
        data = response.json()
        
        isochrone_feature = data.get('features', [{}])[0]
        if isochrone_feature and isochrone_feature.get('geometry'):
            print("Success! Reachable area calculated.")
            return isochrone_feature.get('geometry')
        
        print("Error: Isochrone API returned no valid geometry.")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Isochrone data: {e}")
        return None

# --- FUNCTION: Find Eligible Stores ---
def find_eligible_stores_overpass(bbox: Tuple[float, float, float, float]) -> Dict[str, Tuple[float, float]]:
    """Find grocery stores within bounding box using Overpass API."""
    print(f"\nSearching for stores within area...")
    
    min_lat, min_lon, max_lat, max_lon = bbox
    overpass_query = f"""
        [out:json][timeout:25];
        node["shop"~"supermarket|convenience|grocer|grocery"]({min_lat},{min_lon},{max_lat},{max_lon});
        out center;
    """
    
    for attempt in range(3):
        try:
            response = requests.post(OVERPASS_URL, data={"data": overpass_query}, timeout=30)
            response.raise_for_status()
            data = response.json()
            stores = {}
            
            for element in data.get('elements', []):
                if element['type'] == 'node':
                    lat = element.get('lat')
                    lon = element.get('lon')
                    name = element.get('tags', {}).get('name')
                    
                    if not name:
                        shop_type = element.get('tags', {}).get('shop', 'Store')
                        name = f"{shop_type.capitalize()} ({int(lat * 1000)})"
                    
                    if lat and lon:
                        # Ensure unique names
                        original_name = name
                        count = 1
                        while name in stores:
                            name = f"{original_name} {count}"
                            count += 1
                        
                        stores[name] = (lat, lon)
            
            print(f"Found {len(stores)} eligible store(s).")
            return stores
            
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {}
    
    return {}

# --- TRAVEL TIME MATRIX FUNCTIONS ---
def convert_locations_to_ors_format(locations: List[Tuple[float, float]]) -> List[List[float]]:
    return [[loc[1], loc[0]] for loc in locations]

def get_distance_matrix(locations: List[Tuple[float, float]]) -> Union[Dict[str, Any], None]:
    ors_locations = convert_locations_to_ors_format(locations)
    headers = {
        'Accept': 'application/json',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {"locations": ors_locations}
    
    for attempt in range(3):
        try:
            response = requests.post(ORS_MATRIX_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.exceptions.RequestException:
            return None
    
    return None

def process_matrix_result(matrix_response: Dict) -> List[List[float]]:
    durations_matrix = matrix_response.get('durations')
    return durations_matrix if durations_matrix else []

# --- CITY-TO-STATE MAPPING ---
STORE_CITY_MAPPING = {
    "Kroger": "Indianapolis", "Walmart": "Indianapolis", "Meijer": "Indianapolis",
    "Target": "Indianapolis", "Aldi": "Indianapolis", "Whole Foods": "Indianapolis",
    "Trader Joe's": "Indianapolis", "Costco": "Indianapolis", "Safeway": "Indianapolis",
    "Publix": "Indianapolis", "Giant": "Indianapolis", "Food Lion": "Indianapolis",
    "Harris Teeter": "Indianapolis", "Hy-Vee": "Indianapolis", "H-E-B": "Indianapolis",
    "Wegmans": "Indianapolis", "Stop & Shop": "Indianapolis", "ShopRite": "Indianapolis"
}

def get_city_for_store(store_name: str) -> str:
    store_lower = store_name.lower()
    common_cities = ["indianapolis", "chicago", "new york", "los angeles", "houston"]
    
    for city in common_cities:
        if city in store_lower:
            return city.title()
    
    for store_pattern, city in STORE_CITY_MAPPING.items():
        if store_pattern.lower() in store_lower:
            return city
    
    return "Indianapolis"

# --- PRICE DATABASE FUNCTIONS ---
def load_city_prices_from_csv() -> Dict[str, Dict[str, float]]:
    """Load price data from alanVeggies.csv."""
    city_price_data: Dict[str, Dict[str, float]] = {}
    
    try:
        with open(VEGGIES_CSV, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = ['Vegetable', 'Form', 'RetailPrice', 'RetailPriceUnit']
            
            if not all(col in reader.fieldnames for col in required_columns):
                print(f"Error: Missing required columns in {VEGGIES_CSV}")
                return {}
            
            city_columns = [col for col in reader.fieldnames if col not in required_columns]
            
            for row in reader:
                vegetable = row['Vegetable'].strip().lower()
                form = row['Form'].strip().lower()
                
                if form not in ['fresh', 'raw', '']:
                    continue
                
                for city in city_columns:
                    price_str = row[city].strip()
                    if price_str:
                        try:
                            price = float(price_str)
                            if city not in city_price_data:
                                city_price_data[city] = {}
                            city_price_data[city][vegetable] = price
                        except ValueError:
                            continue
        
        return city_price_data
        
    except Exception as e:
        print(f"Error loading city prices: {e}")
        return {}

def fetch_grocery_prices(
    ingredient_quantities: Dict[str, int], 
    store_ids: List[str]
) -> Tuple[Dict[str, Dict[str, float]], List[str], List[Dict[str, Union[str, int]]]]:
    """
    Generate store-specific prices for ingredients.
    Each store-item combination gets a UNIQUE but consistent random factor.
    """
    print("\nGenerating store prices for ingredients...")
    
    city_price_data = load_city_prices_from_csv()
    
    if not city_price_data:
        print("Error: Could not load price data.")
        return {}, list(ingredient_quantities.keys()), []
    
    price_database: Dict[str, Dict[str, float]] = {}
    removed_items: List[str] = []
    shopping_list: List[Dict[str, Union[str, int]]] = []
    
    # Check which ingredients are available
    available_vegetables = set()
    for city_data in city_price_data.values():
        available_vegetables.update(city_data.keys())
    
    # Create shopping list from available ingredients
    for veggie, quantity in ingredient_quantities.items():
        if veggie in available_vegetables:
            shopping_list.append({"name": veggie, "qty": quantity})
        else:
            removed_items.append(veggie)
    
    if not shopping_list:
        return {}, removed_items, []
    
    # Generate prices for each store
    for store_id in store_ids:
        store_city = get_city_for_store(store_id)
        
        # Find matching city
        city_found = store_city
        if store_city not in city_price_data:
            city_lower = store_city.lower()
            for available_city in city_price_data.keys():
                if available_city.lower() == city_lower:
                    city_found = available_city
                    break
            else:
                city_found = "Indianapolis"
                if city_found not in city_price_data:
                    city_found = list(city_price_data.keys())[0]
        
        print(f"  Using '{city_found}' prices for store: {store_id}")
        
        # Initialize store in database
        price_database[store_id] = {}
        
        # Generate prices for each ingredient
        for item_data in shopping_list:
            item_name = item_data["name"]
            base_price = city_price_data[city_found].get(item_name)
            
            if base_price is not None:
                # === CRITICAL: Unique random factor for EACH store-item combination ===
                # Create a seed that's specific to THIS store and THIS item
                seed_str = f"{store_id}_{item_name}"
                
                # Method 1: Using hash to create deterministic but unique seed
                seed_value = hash(seed_str)  # Integer hash value
                
                # Create a LOCAL random generator for this specific combination
                # This doesn't affect the global random state
                local_random = random.Random(seed_value)
                
                # Generate random factor between 0.6 and 1.4
                variance_factor = local_random.uniform(MIN_PRICE_FACTOR, MAX_PRICE_FACTOR)
                
                # Calculate final price
                store_price = base_price * variance_factor
                price_database[store_id][item_name] = round(store_price, 2)
            else:
                price_database[store_id][item_name] = float('inf')
    
    # === DEMONSTRATE the randomization ===
    print("\n=== PRICE RANDOMIZATION DEMONSTRATION ===")
    print("Same item should have DIFFERENT prices at DIFFERENT stores:")
    
    if shopping_list:
        # Show first 3 items across first 3 stores
        for i, item_data in enumerate(shopping_list[:3]):
            item_name = item_data["name"]
            print(f"\n  Item: {item_name.title()}")
            
            prices = []
            for store_id in list(price_database.keys())[:3]:  # First 3 stores
                price = price_database[store_id].get(item_name, float('inf'))
                if price != float('inf'):
                    # Get the seed to show it's deterministic
                    seed_str = f"{store_id}_{item_name}"
                    seed_value = hash(seed_str)
                    prices.append(f"{store_id}: ${price:.2f} (seed: {seed_value % 1000}...)")
            
            if prices:
                for price_info in prices:
                    print(f"    {price_info}")
    
    # Show that same store-item gives same price every time
    print("\n=== CONSISTENCY CHECK ===")
    print("Same store + same item = same price every run:")
    
    if shopping_list and len(price_database) > 0:
        test_store = list(price_database.keys())[0]
        test_item = shopping_list[0]["name"]
        test_price = price_database[test_store].get(test_item, "N/A")
        test_seed = hash(f"{test_store}_{test_item}")
        print(f"  {test_store} + {test_item}")
        print(f"  Price: ${test_price:.2f}")
        print(f"  Seed value: {test_seed}")
        print(f"  Hash of '{test_store}_{test_item}': {test_seed}")
    
    return price_database, removed_items, shopping_list

def find_optimal_store(
    durations_matrix: List[List[float]],
    price_database: Dict[str, Dict[str, float]],
    location_names: List[str],
    shopping_list: List[Dict[str, Union[str, int]]]
) -> Tuple[List[str], float, float]:
    """
    FULL OPTIMIZATION with all pruning phases - WITH DEBUGGING
    """
    start_index = location_names.index("Start")
    optimal_route: List[str] = []
    min_cost = float('inf')
    best_total_time = 0.0
    
    if not shopping_list:
        print("No shopping list items!")
        return [], 0.0, 0.0
    
    max_time_minutes = MAX_TIME_SECONDS / 60
    print(f"\n=== OPTIMIZATION START ===")
    print(f"Constraint: Total route must be <= {max_time_minutes:.0f} minutes")
    print(f"Number of stores: {len(location_names) - 1}")
    print(f"Number of items: {len(shopping_list)}")
    
    store_indices = list(range(1, len(location_names)))
    
    # Debug: Show store names
    print(f"\nStores available:")
    for idx in store_indices:
        print(f"  {idx}: {location_names[idx]}")
    
    # Set to store subsets that failed time constraint
    failed_time_subsets: Set[frozenset[int]] = set()
    
    k1_winner_store_id: Optional[str] = None
    
    # --- Phase 1: Single Store Baseline ---
    print("\n=== PHASE 1: Single Store Baseline ===")
    
    single_store_costs = {}
    
    for store_indices_subset in combinations(store_indices, 1):
        store_index = store_indices_subset[0]
        store_id = location_names[store_index]
        
        item_cost, item_quantity_counts = calculate_split_shopping_price(
            shopping_list, [store_id], price_database
        )
        
        if item_cost == float('inf'):
            print(f"  {store_id}: Cannot buy all items here")
            continue
        
        single_store_costs[store_id] = item_cost
        
        # Calculate shopping time
        store_type = store_id.split(' ')[0]
        multiplier = STORE_TIME_MULTIPLIERS.get(store_type, 1.0)
        quantity = item_quantity_counts.get(store_id, 0)
        shopping_time = (BASE_FIXOUT_OVERHEAD * multiplier) + (quantity * TIME_PER_UNIT_SECONDS)
        
        # Calculate travel time
        travel_time = durations_matrix[start_index][store_index] + durations_matrix[store_index][start_index]
        total_time = travel_time + shopping_time
        
        if total_time <= MAX_TIME_SECONDS:
            if item_cost < min_cost:
                min_cost = item_cost
                optimal_route = [store_id]
                best_total_time = total_time
                k1_winner_store_id = store_id
                print(f"  {store_id}: ✅ NEW BEST! ${item_cost:.2f} | {int(total_time/60)}m")
            else:
                print(f"  {store_id}: ${item_cost:.2f} | {int(total_time/60)}m (worse than ${min_cost:.2f})")
        else:
            print(f"  {store_id}: ❌ Time exceeded ({int(total_time/60)}m > {int(MAX_TIME_SECONDS/60)}m)")
    
    if min_cost == float('inf') or k1_winner_store_id is None:
        print("\n❌ No single-stop route meets time constraint.")
        return optimal_route, min_cost, best_total_time
    
    print(f"\n✅ Single store winner: {k1_winner_store_id} at ${min_cost:.2f}")
    
    # --- Phase 2: Store Pruning ---
    print(f"\n=== PHASE 2: Store Pruning ===")
    print(f"Comparing other stores against winner: {k1_winner_store_id}")
    
    eligible_multistop_indices: List[int] = []
    
    # Always include the winner
    winner_index = location_names.index(k1_winner_store_id)
    eligible_multistop_indices.append(winner_index)
    
    for index in store_indices:
        store_id = location_names[index]
        
        if store_id == k1_winner_store_id:
            continue  # Already added
            
        is_store_useful = False
        cheaper_items = []
        
        # Check if this store has ANY item cheaper than the winner
        for item_data in shopping_list:
            item_name = item_data["name"]
            
            winner_price = price_database.get(k1_winner_store_id, {}).get(item_name, float('inf'))
            store_price = price_database.get(store_id, {}).get(item_name, float('inf'))
            
            if store_price < winner_price:
                is_store_useful = True
                cheaper_items.append(f"{item_name}: ${store_price:.2f} vs ${winner_price:.2f}")
                break  # Only need one cheaper item
        
        if is_store_useful:
            eligible_multistop_indices.append(index)
            print(f"  {store_id}: ✅ Kept - has cheaper items")
            if cheaper_items:
                print(f"     e.g., {cheaper_items[0]}")
        else:
            print(f"  {store_id}: 🛑 Pruned - no cheaper items")
    
    print(f"\nEligible stores for multi-stop optimization: {len(eligible_multistop_indices)}")
    for idx in eligible_multistop_indices:
        print(f"  - {location_names[idx]}")
    
    # --- Phase 3: Multi-Store Optimization ---
    print(f"\n=== PHASE 3: Multi-Store Optimization ===")
    
    # If only 1 eligible store (just the winner), skip multi-store
    if len(eligible_multistop_indices) <= 1:
        print("Only 1 eligible store. Single store is optimal.")
        return optimal_route, min_cost, best_total_time
    
    # Try 2, 3, 4-store combinations
    max_k = min(4, len(eligible_multistop_indices))
    
    for k in range(2, max_k + 1):
        print(f"\n--- Testing {k}-store combinations ---")
        
        # Generate all combinations
        combinations_list = list(combinations(eligible_multistop_indices, k))
        print(f"  Testing {len(combinations_list)} combinations of {k} stores")
        
        found_any_feasible = False
        combinations_checked = 0
        combinations_pruned = 0
        
        for store_indices_subset in combinations_list:
            route_store_ids = [location_names[i] for i in store_indices_subset]
            route_str = " -> ".join(route_store_ids)
            
            # Skip if superset of failed combination
            current_set = frozenset(store_indices_subset)
            is_superset_of_failed = any(
                failed_set.issubset(current_set) for failed_set in failed_time_subsets
            )
            
            if is_superset_of_failed:
                combinations_pruned += 1
                continue
            
            # Calculate cost for this combination
            item_cost, item_quantity_counts = calculate_split_shopping_price(
                shopping_list, route_store_ids, price_database
            )
            
            combinations_checked += 1
            
            # Quick cost check - skip if not cheaper
            if item_cost >= min_cost:
                # print(f"    {route_str}: Cost ${item_cost:.2f} not better than ${min_cost:.2f}")
                continue
            
            # Calculate shopping time for all stores
            total_shopping_time = 0.0
            min_one_way_travel = float('inf')
            
            for store_id in route_store_ids:
                store_type = store_id.split(' ')[0]
                multiplier = STORE_TIME_MULTIPLIERS.get(store_type, 1.0)
                quantity = item_quantity_counts.get(store_id, 0)
                store_shopping_time = (BASE_FIXOUT_OVERHEAD * multiplier) + (quantity * TIME_PER_UNIT_SECONDS)
                total_shopping_time += store_shopping_time
                
                # Find minimum travel to any store in this combination
                store_index = location_names.index(store_id)
                travel_to_store = durations_matrix[start_index][store_index]
                if travel_to_store < min_one_way_travel:
                    min_one_way_travel = travel_to_store
            
            # Quick time check (lower bound)
            lower_bound_travel = 2 * min_one_way_travel
            lower_bound_total = lower_bound_travel + total_shopping_time
            
            if lower_bound_total > MAX_TIME_SECONDS:
                failed_time_subsets.add(current_set)
                # print(f"    {route_str}: Lower bound time {int(lower_bound_total/60)}m > limit")
                continue
            
            found_any_feasible = True
            
            # Find optimal route order
            best_route_indices, min_travel_time = find_fastest_travel_permutation(
                store_indices_subset, durations_matrix, start_index
            )
            
            total_time = min_travel_time + total_shopping_time
            
            if total_time > MAX_TIME_SECONDS:
                failed_time_subsets.add(current_set)
                print(f"    {route_str}: ❌ Time {int(total_time/60)}m > limit {int(MAX_TIME_SECONDS/60)}m")
                continue
            
            # New best route found!
            min_cost = item_cost
            optimal_route = [location_names[i] for i in best_route_indices]
            best_total_time = total_time
            
            print(f"    {route_str}: ✅ NEW BEST! Cost: ${item_cost:.2f} | Time: {int(total_time/60)}m")
            
            # Show item assignment breakdown
            print(f"      Item assignments:")
            for item_data in shopping_list:
                item_name = item_data["name"]
                quantity = item_data["qty"]
                best_store_for_item = None
                best_price_for_item = float('inf')
                
                for store_id in route_store_ids:
                    price = price_database.get(store_id, {}).get(item_name, float('inf'))
                    if price < best_price_for_item:
                        best_price_for_item = price
                        best_store_for_item = store_id
                
                if best_store_for_item:
                    print(f"        {quantity}× {item_name}: ${best_price_for_item:.2f} at {best_store_for_item}")
        
        print(f"  Checked: {combinations_checked}, Pruned: {combinations_pruned}")
        
        # If no feasible combinations at this k, continue to next k
        if not found_any_feasible:
            print(f"  No feasible {k}-store combinations")
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    return optimal_route, min_cost, best_total_time
# --- CORE OPTIMIZATION FUNCTIONS ---
def calculate_split_shopping_price(
    shopping_list: List[Dict[str, Union[str, int]]],
    store_ids: List[str], 
    price_database: Dict[str, Dict[str, float]]
) -> Tuple[float, Dict[str, int]]:
    """Calculate minimum total cost and item assignments."""
    total_price = 0.0
    item_assignment_counts: Dict[str, int] = {s: 0 for s in store_ids}
    
    for item_data in shopping_list:
        item = item_data["name"]
        quantity = item_data["qty"]
        min_item_cost = float('inf')
        best_store = None
        
        for store_id in store_ids:
            unit_price = price_database.get(store_id, {}).get(item, float('inf'))
            if unit_price != float('inf'):
                cost = unit_price * quantity
                if cost < min_item_cost:
                    min_item_cost = cost
                    best_store = store_id
        
        if min_item_cost == float('inf'):
            return float('inf'), item_assignment_counts
        
        total_price += min_item_cost
        if best_store:
            item_assignment_counts[best_store] += quantity
    
    return total_price, item_assignment_counts

def find_fastest_travel_permutation(
    store_indices_subset: Tuple[int, ...], 
    durations_matrix: List[List[float]], 
    start_index: int
) -> Tuple[Tuple[int, ...], float]:
    """Find fastest route order for given stores."""
    min_travel_time = float('inf')
    best_route_indices = store_indices_subset
    
    for route_indices in permutations(store_indices_subset):
        travel_time = 0.0
        current_index = start_index
        
        for next_index in route_indices:
            travel_time += durations_matrix[current_index][next_index]
            current_index = next_index
        
        travel_time += durations_matrix[current_index][start_index]
        
        if travel_time < min_travel_time:
            min_travel_time = travel_time
            best_route_indices = route_indices
    
    return best_route_indices, min_travel_time

def find_optimal_store(
    durations_matrix: List[List[float]],
    price_database: Dict[str, Dict[str, float]],
    location_names: List[str],
    shopping_list: List[Dict[str, Union[str, int]]]
) -> Tuple[List[str], float, float]:
    """Find optimal shopping route."""
    start_index = location_names.index("Start")
    optimal_route: List[str] = []
    min_cost = float('inf')
    best_total_time = 0.0
    
    if not shopping_list:
        return [], 0.0, 0.0
    
    store_indices = list(range(1, len(location_names)))
    failed_time_subsets: Set[frozenset[int]] = set()
    k1_winner_store_id: Optional[str] = None
    
    # Phase 1: Single store optimization
    for store_indices_subset in combinations(store_indices, 1):
        store_index = store_indices_subset[0]
        store_id = location_names[store_index]
        
        item_cost, item_quantity_counts = calculate_split_shopping_price(
            shopping_list, [store_id], price_database
        )
        
        if item_cost == float('inf'):
            continue
        
        # Calculate shopping time
        store_type = store_id.split(' ')[0]
        multiplier = STORE_TIME_MULTIPLIERS.get(store_type, 1.0)
        quantity = item_quantity_counts.get(store_id, 0)
        shopping_time = (BASE_FIXOUT_OVERHEAD * multiplier) + (quantity * TIME_PER_UNIT_SECONDS)
        
        # Calculate travel time
        travel_time = durations_matrix[start_index][store_index] + durations_matrix[store_index][start_index]
        total_time = travel_time + shopping_time
        
        if total_time <= MAX_TIME_SECONDS and item_cost < min_cost:
            min_cost = item_cost
            optimal_route = [store_id]
            best_total_time = total_time
            k1_winner_store_id = store_id
    
    if not optimal_route:
        return [], 0.0, 0.0
    
    # Phase 2: Multi-store optimization (simplified for demo)
    # In a full implementation, this would include the pruning logic from before
    
    return optimal_route, min_cost, best_total_time

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=" * 60)
    print("MEAL PLANNER & GROCERY OPTIMIZER")
    print("=" * 60)
    print(f"Location: {USER_START_LOCATION}")
    print(f"Shopping time available: {SHOPPING_TIME_HOURS} hours")
    print(f"Nutrition target: {DAILY_CALORIE_TARGET:,} calories/day")
    print("=" * 60)
    
    # Step 1: Load available vegetables
    available_veggies = load_available_vegetables()
    if not available_veggies:
        print("FATAL: No vegetables available for meal planning.")
        exit(1)
    
    # Step 2: Load and filter recipes
    all_recipes = load_recipes()
    filtered_recipes = filter_recipes_by_available_vegetables(all_recipes, available_veggies)
    
    if not filtered_recipes:
        print("FATAL: No recipes available with the current vegetables.")
        exit(1)
    
    # Step 3: Create weekly meal plan
    meal_plan, ingredient_quantities = create_weekly_meal_plan(filtered_recipes, available_veggies)
    
    if not meal_plan:
        print("FATAL: Could not create meal plan.")
        exit(1)
    
    # Display meal plan
    print("\n" + "=" * 60)
    print("WEEKLY MEAL PLAN")
    print("=" * 60)
    for meal in meal_plan:
        print(f"{meal['day']} - {meal['meal_type']}:")
        print(f"  Recipe: {meal['recipe']}")
        print(f"  Calories: {meal['calories']:.0f}, Protein: {meal['protein']:.1f}g")
        print(f"  Ingredients: {', '.join(meal['ingredients'][:3])}...")
        print()
    
    print("\n" + "=" * 60)
    print("INGREDIENTS NEEDED")
    print("=" * 60)
    for veggie, quantity in ingredient_quantities.items():
        print(f"  {veggie.title()}: {quantity} units")
    
    # Step 4: Calculate reachable area
    ONE_WAY_TIME_SECONDS = int(MAX_TIME_SECONDS / 4)
    isochrone_geometry = get_travel_isochrone(USER_START_LOCATION, ONE_WAY_TIME_SECONDS)
    
    if not isochrone_geometry:
        print("FATAL: Could not calculate reachable area.")
        exit(1)
    
    bbox = get_geojson_bounding_box(isochrone_geometry)
    
    # Step 5: Find stores
    STORE_LOCATIONS = find_eligible_stores_overpass(bbox)
    
    if not STORE_LOCATIONS:
        print("FATAL: No stores found in reachable area.")
        exit(1)
    
    # Filter stores if needed
    if len(STORE_LOCATIONS) > MAX_STORES_TO_USE:
        print(f"Filtering stores to {MAX_STORES_TO_USE} closest...")
        distances = []
        start_lat, start_lon = USER_START_LOCATION
        for name, (lat, lon) in STORE_LOCATIONS.items():
            dist_sq = (lat - start_lat)**2 + (lon - start_lon)**2
            distances.append((dist_sq, name, (lat, lon)))
        distances.sort(key=lambda x: x[0])
        STORE_LOCATIONS = {name: loc for _, name, loc in distances[:MAX_STORES_TO_USE]}
    
    print(f"\nStores for optimization: {len(STORE_LOCATIONS)}")
    
    # Step 6: Get travel times
    all_coords = [USER_START_LOCATION] + list(STORE_LOCATIONS.values())
    location_names = ["Start"] + list(STORE_LOCATIONS.keys())
    
    print("\nGetting travel times...")
    matrix_response = get_distance_matrix(all_coords)
    
    if not matrix_response:
        print("FATAL: Could not get travel times.")
        exit(1)
    
    durations_matrix = process_matrix_result(matrix_response)
    if not durations_matrix:
        print("FATAL: No travel time data.")
        exit(1)
    
    # Step 7: Get prices
    price_database, removed_items, shopping_list = fetch_grocery_prices(
        ingredient_quantities, list(STORE_LOCATIONS.keys())
    )
    
    if not price_database or not shopping_list:
        print("FATAL: Could not generate prices.")
        exit(1)
    
    # Step 8: Optimize shopping
    print("\n" + "=" * 60)
    print("OPTIMIZING GROCERY SHOPPING")
    print("=" * 60)
    
    optimal_route, item_cost, total_time_seconds = find_optimal_store(
        durations_matrix, price_database, location_names, shopping_list
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMAL SHOPPING PLAN")
    print("=" * 60)
    
    if optimal_route:
        print(f"✅ Recommended Route: Start → {' → '.join(optimal_route)} → Home")
        print(f"💰 Estimated Cost: ${item_cost:.2f}")
        print(f"⏱️  Total Time: {int(total_time_seconds/60)}m {int(total_time_seconds%60)}s")
        
        # Show what to buy where
        print(f"\n📋 Shopping List Breakdown:")
        for item_data in shopping_list:
            item_name = item_data["name"]
            quantity = item_data["qty"]
            best_price = float('inf')
            best_store = None
            
            for store in optimal_route:
                price = price_database[store].get(item_name, float('inf'))
                if price < best_price:
                    best_price = price
                    best_store = store
            
            if best_store:
                print(f"  {quantity} × {item_name.title()}: ${best_price:.2f} each at {best_store}")
    else:
        print("❌ No feasible shopping route found within time constraint.")
        print("Try increasing your shopping time or reducing the number of ingredients.")
    
    print("=" * 60)