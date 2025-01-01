import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import random
import itertools

# Load GTFS data
stops = pd.read_csv("stops.csv")
stop_times = pd.read_csv("stop_times.csv")
routes = pd.read_csv("routes.csv")
trips = pd.read_csv("trips.csv")
routes_with_pois = pd.read_csv("routes_with_pois.csv")
trips_with_pois = pd.read_csv("trips_with_pois.csv")
transfers = pd.read_csv("transfers.csv")

# Initialize the graph
graph = nx.Graph()

# Fee calculation function based on distance
def calculate_dynamic_fee(distance):
    min_fee = 0.20
    max_fee = 0.50
    max_distance = 10000  # meters (10 km as an example threshold)
    return min_fee + (max_fee - min_fee) * min(distance, max_distance) / max_distance

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

        # Calculate dynamic fee based on distance
        fee = calculate_dynamic_fee(distance)

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
            pheromone=1.0,    # Initial pheromone level for ACO
            visibility=1.0 / distance if distance > 0 else 0,  # Inverse of distance
            trip_id=trip_id,
            route_id=route_id,
            fee=fee,  # Dynamic fee based on distance
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

# Ant Colony Optimization for Route Finding
def ant_colony_route(graph, source, target, user_prefs, iterations=10, ants=5, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    def heuristic(edge_data):
        distance = edge_data.get('weight', 0)
        fee = edge_data.get('fee', 0)
        scenic_score = edge_data.get('scenic_score', 0)
        transfers = 1 if edge_data.get('is_transfer', False) else 0
        return calculate_combined_weight(distance, fee, scenic_score, transfers, 0, user_prefs)

    best_path = None
    best_path_cost = float('inf')

    for _ in range(iterations):
        paths = []
        path_costs = []

        for _ in range(ants):
            path = [source]
            current = source
            visited = set()

            while current != target:
                visited.add(current)
                neighbors = [n for n in graph.neighbors(current) if n not in visited]
                if not neighbors:
                    break

                probabilities = []
                for neighbor in neighbors:
                    edge_data = graph[current][neighbor]
                    pheromone = edge_data['pheromone'] ** alpha
                    vis = heuristic(edge_data) ** beta
                    probabilities.append(pheromone * vis)

                probabilities_sum = sum(probabilities)
                probabilities = [p / probabilities_sum for p in probabilities]

                next_stop = random.choices(neighbors, probabilities)[0]
                path.append(next_stop)
                current = next_stop

            if current == target:
                paths.append(path)
                cost = sum(heuristic(graph[u][v]) for u, v in zip(path[:-1], path[1:]))
                path_costs.append(cost)

        if paths:
            best_index = path_costs.index(min(path_costs))
            if path_costs[best_index] < best_path_cost:
                best_path = paths[best_index]
                best_path_cost = path_costs[best_index]

        # Update pheromones
        for u, v in graph.edges():
            graph[u][v]['pheromone'] *= (1 - evaporation_rate)

        for path, cost in zip(paths, path_costs):
            for u, v in zip(path[:-1], path[1:]):
                graph[u][v]['pheromone'] += 1.0 / cost

    return best_path, best_path_cost

# Testing multiple parameter configurations
# Define ranges for parameters
ants_range = [5, 10, 20]
iterations_range = [10, 50, 100]
alpha_range = [0.5, 1.0, 1.5]
beta_range = [2.0, 3.0, 4.0]
evaporation_rate_range = [0.1, 0.3, 0.5]

# Results storage
results = []

# Source and target nodes
source = "AG10" # Example source
target = "PY04" # Example target

# User preferences
user_prefs = {
    'distance': 1.0,
    'cost': 1.0,
    'scenic': 1.0,
    'comfort': 1.0,
    'disabled_friendly': 0.0,
}

# Loop over all combinations of parameters
for ants, iterations, alpha, beta, evaporation_rate in itertools.product(
    ants_range, iterations_range, alpha_range, beta_range, evaporation_rate_range
):
    try:
        best_path, best_cost = ant_colony_route(
            graph, source, target, user_prefs,
            iterations=iterations,
            ants=ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate
        )

        if best_path:
            # Calculate performance metrics
            total_distance = sum(graph[u][v]['weight'] for u, v in zip(best_path[:-1], best_path[1:])) / 1000  # km
            total_time = sum(graph[u][v].get('time_taken', 0) for u, v in zip(best_path[:-1], best_path[1:])) / 60  # minutes
            total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(best_path[:-1], best_path[1:]))
            interchanges = sum(1 for u, v in zip(best_path[:-1], best_path[1:]) if graph[u][v].get('is_transfer', False))

            results.append({
                'ants': ants,
                'iterations': iterations,
                'alpha': alpha,
                'beta': beta,
                'evaporation_rate': evaporation_rate,
                'best_cost': best_cost,
                'total_distance': total_distance,
                'total_time': total_time,
                'total_cost': total_cost,
                'interchanges': interchanges,
                'best_path': best_path
            })
    except Exception as e:
        print(f"Error with parameters ants={ants}, iterations={iterations}, alpha={alpha}, beta={beta}, evaporation_rate={evaporation_rate}: {e}")

# Save results to a DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['best_cost', 'total_distance', 'total_time'])

# Save to CSV
results_df.to_csv("ant_colony_optimization_results.csv", index=False)

# Display top results
import ace_tools_open as tools; tools.display_dataframe_to_user(name="Ant Colony Optimization Results", dataframe=results_df)
