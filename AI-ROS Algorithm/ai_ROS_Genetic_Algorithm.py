import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Load GTFS data
stops = pd.read_csv("Original GTFS Data/stops.csv")
stop_times = pd.read_csv("Original GTFS Data/stop_times.csv")
routes = pd.read_csv("Original GTFS Data/routes.csv")
trips = pd.read_csv("Original GTFS Data/trips.csv")
routes_with_pois = pd.read_csv("GTFS Data Creation/routes_with_pois.csv")
trips_with_pois = pd.read_csv("GTFS Data Creation/trips_with_pois.csv")
transfers = pd.read_csv("GTFS Data Creation/transfers.csv")

# Initialize the graph
graph = nx.Graph()

# Fee calculation function based on distance
def calculate_dynamic_fare_with_increment(distance):
    initial_fare = 0.20  # Initial starting price
    max_increment = 0.50  # Maximum increment for longest distances
    max_distance = 10000  # meters (10 km as an example threshold)

    # Calculate increment proportionally
    increment = (max_increment - 0.10) * min(distance, max_distance) / max_distance + 0.10
    return initial_fare + increment


# Helper function to calculate time difference in seconds
def calculate_time_difference(time1, time2):
    time_format = "%H:%M:%S"
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)
    return (t2 - t1).total_seconds()

# Add nodes (stops)
for _, stop in stops.iterrows():
    graph.add_node(
        stop['stop_id'],  # Node ID
        stop_name=stop['stop_name'],
        location=(stop['stop_lat'], stop['stop_lon']),
        category=stop.get('category', None),
        isOKU=stop.get('isOKU', None),
        nearby_pois=stop.get('nearby_pois', '[]'),  # Add POIs if available
        search_name=stop.get('search', ''),  # Store search column
        additional_data=stop.to_dict()  # Store all stop attributes
    )

# Map trips and routes for easy access
trip_route_map = trips.set_index('trip_id')['route_id'].to_dict()
route_data_map = routes.set_index('route_id').to_dict('index')

# Add edges (connections between stops)
previous_stop = None
previous_departure_time = None
current_trip_id = None

for _, stop_time in stop_times.iterrows():
    trip_id = stop_time['trip_id']
    stop_id = stop_time['stop_id']
    arrival_time = stop_time['arrival_time']
    departure_time = stop_time['departure_time']

    if trip_id != current_trip_id:
        # New trip; reset previous stop and time
        previous_stop = None
        previous_departure_time = None
        current_trip_id = trip_id

    if previous_stop:
        # Add edge between previous stop and current stop
        stop1_location = (
            graph.nodes[previous_stop]['location'][0],
            graph.nodes[previous_stop]['location'][1]
        )
        stop2_location = (
            graph.nodes[stop_id]['location'][0],
            graph.nodes[stop_id]['location'][1]
        )
        distance = geodesic(stop1_location, stop2_location).meters

        # Calculate dynamic fare based on the new logic
        fee = calculate_dynamic_fare_with_increment(distance)

        # Calculate time taken from stop_times data
        time_taken = calculate_time_difference(previous_departure_time, arrival_time)

        # Define scenic score
        scenic_score = len(graph.nodes[previous_stop].get('nearby_pois', [])) + len(graph.nodes[stop_id].get('nearby_pois', []))

        # Add route and trip metadata
        route_id = trip_route_map.get(trip_id)
        route_data = route_data_map.get(route_id, {})

        graph.add_edge(
            previous_stop,
            stop_id,
            weight=distance,  # Add distance as weight
            pheromone=1.0,    # Initial pheromone level for GA
            visibility=1.0 / distance if distance > 0 else 0,  # Inverse of distance
            trip_id=trip_id,
            route_id=route_id,
            fee=fee,  # Dynamic fee based on new logic
            time_taken=time_taken,  # Time taken in seconds
            scenic_score=scenic_score,  # Add scenic score
            route_data=route_data,  # Add route metadata
            schedule_data=stop_time.to_dict()  # Store all stop_time attributes
        )

    previous_stop = stop_id
    previous_departure_time = departure_time

# Add transfer edges from transfer.csv
for _, transfer in transfers.iterrows():
    graph.add_edge(
        transfer['from_stop_id'],
        transfer['to_stop_id'],
        weight=transfer['min_transfer_time'],  # Use transfer time as weight
        transfer_type=transfer['transfer_type'],
        min_transfer_time=transfer['min_transfer_time'],
        pheromone=1.0,  # Add default pheromone level
        visibility=1.0 / transfer['min_transfer_time'] if transfer['min_transfer_time'] > 0 else 0,  # Inverse of time
        is_transfer=True,  # Mark as transfer edge
        time_taken=transfer['min_transfer_time'],  # Set default time_taken for transfers
        fee=0.0  # Default fee for transfers
    )

# Add routes and trips data as graph attributes
for _, route in routes_with_pois.iterrows():
    graph.graph[route['route_id']] = {
        'route_pois': route.get('route_pois', []),
        'additional_data': route.to_dict()
    }

for _, trip in trips_with_pois.iterrows():
    graph.graph[trip['trip_id']] = {
        'trip_pois': trip.get('trip_pois', []),
        'additional_data': trip.to_dict()
    }

# Function to apply user preferences dynamically
def calculate_combined_weight(distance, fee, scenic_score, transfers, isOKU, user_prefs):
    weight = (
        user_prefs['distance'] * distance +
        user_prefs['cost'] * fee -
        user_prefs['scenic'] * scenic_score +
        user_prefs['comfort'] * transfers -
        user_prefs['disabled_friendly'] * (1 if isOKU else 0)
    )
    return weight

# Genetic Algorithm for Route Finding
def genetic_algorithm_route(graph, source, target, user_prefs, population_size=50, generations=100, mutation_rate=0.1):
    def is_valid_path(path, graph):
        for u, v in zip(path[:-1], path[1:]):
            if not graph.has_edge(u, v):
                return False
        return True

    def fitness(path):
        try:
            total_distance = 0
            total_fee = 0
            total_scenic_score = 0
            total_transfers = 0
            isOKU = any(graph.nodes[stop].get('isOKU', False) for stop in path)

            for u, v in zip(path[:-1], path[1:]):
                if graph.has_edge(u, v):
                    total_distance += graph[u][v]['weight']
                    total_fee += graph[u][v]['fee']
                    total_scenic_score += graph[u][v]['scenic_score']
                    if graph[u][v].get('is_transfer', False):
                        total_transfers += 1
                else:
                    return float('-inf')  # Penalize invalid paths

            return -calculate_combined_weight(total_distance, total_fee, total_scenic_score, total_transfers, isOKU, user_prefs)
        except KeyError:
            return float('-inf')

    def mutate(path):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(path)), 2)
            path[idx1], path[idx2] = path[idx2], path[idx1]
        return path

    def crossover(parent1, parent2):
        cut = random.randint(1, len(parent1) - 2)
        child = parent1[:cut] + [node for node in parent2 if node not in parent1[:cut]]
        return child

    def generate_initial_population():
        population = []
        try:
            base_path = nx.shortest_path(graph, source, target, weight='weight')
            population.append(base_path)  # Always include the shortest path
        except nx.NetworkXNoPath:
            raise ValueError("No shortest path exists between source and target.")

        for _ in range(population_size - 1):
            try:
                path = base_path.copy()
                random.shuffle(path[1:-1])  # Shuffle only intermediate nodes
                if is_valid_path(path, graph):
                    population.append(path)
            except Exception:
                continue
        return population

    population = generate_initial_population()
    if not population:
         print("No valid initial paths found. Falling back to shortest path.")
    try:
        return nx.shortest_path(graph, source, target, weight='weight'), float('inf')
    except nx.NetworkXNoPath:
        raise ValueError("No valid path exists between source and target.")


    best_path = None
    best_fitness = float('-inf')

    for _ in range(generations):
        population = sorted(population, key=lambda p: fitness(p), reverse=True)
        if fitness(population[0]) > best_fitness:
            best_path = population[0]
            best_fitness = fitness(population[0])

        new_population = population[:10]  # Elitism
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:20], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            if is_valid_path(child, graph):
                new_population.append(child)

        population = new_population

    return best_path, -best_fitness

# Diagnostic Checks
# Example: User preferences and route generation
user_prefs = {
    'distance': 1.0,
    'cost': 1.0,
    'scenic': 1.0,
    'comfort': 1.0,  # Added comfortability
    'disabled_friendly': 0.0,
}

source = "AG10"  # PUDU
target = "PY04"  # SUNGAI BULOH

# Check if source and target nodes exist
if source not in graph.nodes:
    print(f"Source node '{source}' not found in the graph.")
if target not in graph.nodes:
    print(f"Target node '{target}' not found in the graph.")

# Check graph connectivity
if nx.is_connected(graph):
    print("Graph is connected.")
else:
    print("Graph is not connected.")

# Check if source and target are in the same connected component
connected_components = list(nx.connected_components(graph))
source_component = next((comp for comp in connected_components if source in comp), None)
target_component = next((comp for comp in connected_components if target in comp), None)

if source_component and target_component and source_component == target_component:
    print("Source and target are in the same connected component.")
else:
    print("Source and target are disconnected.")

# Debugging: Verify shortest path
try:
    shortest_path = nx.shortest_path(graph, source, target, weight='weight')
    print("Shortest path (by weight):", shortest_path)
except nx.NetworkXNoPath:
    print("No valid shortest path exists between the source and target.")

# Generate the best path
try:
    best_path, best_cost = genetic_algorithm_route(graph, source, target, user_prefs)

    if best_path:
        # Calculate ETA, total distance, interchanges, and scenic places
        total_distance = sum(graph[u][v]['weight'] for u, v in zip(best_path[:-1], best_path[1:])) / 1000  # Convert to km
        total_time = sum(graph[u][v].get('time_taken', 0) for u, v in zip(best_path[:-1], best_path[1:])) / 60  # Convert to minutes
        total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(best_path[:-1], best_path[1:]))
        interchanges = sum(1 for u, v in zip(best_path[:-1], best_path[1:]) if graph[u][v].get('is_transfer', False))
        scenic_places = [graph.nodes[stop]['nearby_pois'] for stop in best_path]

        # Convert best_path to station names
        station_names = [graph.nodes[stop]['search_name'] for stop in best_path]

        print(f"Best Path: {' -> '.join(station_names)}")
        print(f"Total Distance: {total_distance:.2f} km")
        print(f"ETA: {total_time:.2f} minutes")
        print(f"Total Cost: RM {total_cost:.2f}")
        print(f"Interchanges: {interchanges}")
        print(f"Scenic Places: {scenic_places}")
    else:
        print("No valid path found.")
except ValueError as e:
    print(e)
