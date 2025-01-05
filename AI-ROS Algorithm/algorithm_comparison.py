import time
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

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

def plot_graph(graph):
    # Get node positions (latitude and longitude)
    pos = {node: (data['location'][1], data['location'][0]) for node, data in graph.nodes(data=True)}

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='blue', alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.5)

    # Draw labels (optional, can be slow for large graphs)
    labels = {node: node for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_color='black')

    # Show the plot
    plt.title("Graph Visualization with Nodes and Edges")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

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
            pheromone=1.0,    # Initial pheromone level for ACO
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

# A* Algorithm for Route Finding
def a_star_route(graph, source, target, user_prefs):
    def heuristic(node, target):
        # Use straight-line distance (haversine) as heuristic
        node_location = graph.nodes[node]['location']
        target_location = graph.nodes[target]['location']
        return geodesic(node_location, target_location).meters

    def path_cost(path):
        total_distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        total_fee = sum(graph[u][v]['fee'] for u, v in zip(path[:-1], path[1:]))
        total_scenic_score = sum(graph[u][v].get('scenic_score', 0) for u, v in zip(path[:-1], path[1:]))  # Default to 0
        total_transfers = sum(1 for u, v in zip(path[:-1], path[1:]) if graph[u][v].get('is_transfer', False))
        isOKU = any(graph.nodes[stop].get('isOKU', False) for stop in path)

        return calculate_combined_weight(total_distance, total_fee, total_scenic_score, total_transfers, isOKU, user_prefs)

    open_set = {source}
    came_from = {}

    g_score = {node: float('inf') for node in graph.nodes}
    g_score[source] = 0

    f_score = {node: float('inf') for node in graph.nodes}
    f_score[source] = heuristic(source, target)

    while open_set:
        current = min(open_set, key=lambda node: f_score[node])

        if current == target:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(source)
            path.reverse()
            return path, path_cost(path)

        open_set.remove(current)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)

                if neighbor not in open_set:
                    open_set.add(neighbor)

    raise ValueError("No valid path found.")

# Ant Colony Optimization for Route Finding
def ant_colony_route(graph, source, target, user_prefs, iterations=50, ants=5, alpha=0.5, beta=3.0, evaporation_rate=0.3):
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

# Dijkstra Algorithm for Route Finding
def dijkstra_route(graph, source, target):
    try:
        path = nx.dijkstra_path(graph, source, target, weight='weight')
        total_distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) / 1000  # Convert to km
        total_time = sum(graph[u][v].get('time_taken', 0) for u, v in zip(path[:-1], path[1:])) / 60  # Convert to minutes
        total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(path[:-1], path[1:]))
        interchanges = sum(1 for u, v in zip(path[:-1], path[1:]) if graph[u][v].get('is_transfer', False))
        scenic_places = [graph.nodes[stop]['nearby_pois'] for stop in path]

        station_names = [graph.nodes[stop]['search_name'] for stop in path]

        print(f"Best Path: {' -> '.join(station_names)}")
        print(f"Total Distance: {total_distance:.2f} km")
        print(f"ETA: {total_time:.2f} minutes")
        print(f"Total Cost: RM {total_cost:.2f}")
        print(f"Interchanges: {interchanges}")
        print(f"Scenic Places: {scenic_places}")
    except nx.NetworkXNoPath:
        print("No valid path found.")

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

# Function to generate varied user preferences
def generate_varied_user_prefs(base_prefs, num_variations, variation_factor=0.1):
    varied_prefs = []
    for _ in range(num_variations):
        new_prefs = base_prefs.copy()
        for key in new_prefs:
            # Add a small random variation to each preference
            new_prefs[key] += random.uniform(-variation_factor, variation_factor)
            # Ensure preferences are non-negative
            new_prefs[key] = max(0, new_prefs[key])
        varied_prefs.append(new_prefs)
    return varied_prefs

# Function to check if routes are unique
def are_routes_unique(routes):
    unique_routes = set()
    for route in routes:
        # Convert the route to a tuple of stop IDs to make it hashable
        route_tuple = tuple(route)
        if route_tuple in unique_routes:
            return False
        unique_routes.add(route_tuple)
    return True

# Function to compare algorithms with varied user preferences
def compare_algorithms_with_varied_prefs(graph, source, target, base_user_prefs, num_runs=10, num_variations=3, variation_factor=0.1):
    algorithms = {
        "A*": a_star_route,
        "Ant Colony Optimization": ant_colony_route,
        "Dijkstra": dijkstra_route,
        "Genetic Algorithm": genetic_algorithm_route
    }

    results = []

    # Generate varied user preferences
    varied_prefs = generate_varied_user_prefs(base_user_prefs, num_variations, variation_factor)

    for algo_name, algo_func in algorithms.items():
        execution_times = []
        path_costs = []
        path_lengths = []
        success_count = 0
        unique_routes = []

        for prefs in varied_prefs:
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    if algo_name == "Dijkstra":
                        # Dijkstra doesn't use user_prefs
                        path = algo_func(graph, source, target)
                        total_distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) / 1000  # Convert to km
                        total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(path[:-1], path[1:]))
                        cost = calculate_combined_weight(total_distance, total_cost, sum(graph[u][v].get('scenic_score', 0) for u, v in zip(path[:-1], path[1:])), sum(1 for u, v in zip(path[:-1], path[1:]) if graph[u][v].get('is_transfer', False)), any(graph.nodes[stop].get('isOKU', False) for stop in path), prefs)
                    else:
                        path, cost = algo_func(graph, source, target, prefs)
                        total_distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) / 1000  # Convert to km
                        total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(path[:-1], path[1:]))

                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    path_costs.append(cost)
                    path_lengths.append(len(path))
                    unique_routes.append(path)
                    success_count += 1
                except Exception as e:
                    print(f"{algo_name} failed in run {_ + 1}: {e}")

        if success_count > 0:
            avg_execution_time = np.mean(execution_times)
            std_execution_time = np.std(execution_times)
            avg_path_cost = np.mean(path_costs)
            avg_path_length = np.mean(path_lengths)
            success_rate = success_count / (num_runs * num_variations)
            unique_route_count = len(set(tuple(route) for route in unique_routes))
        else:
            avg_execution_time = std_execution_time = avg_path_cost = avg_path_length = success_rate = unique_route_count = 0

        results.append({
            "Algorithm": algo_name,
            "Source-Target": f"{source}-{target}",
            "Avg Execution Time": avg_execution_time,
            "Std Execution Time": std_execution_time,
            "Avg Path Cost": avg_path_cost,
            "Avg Path Length": avg_path_length,
            "Success Rate": success_rate,
            "Unique Routes": unique_route_count
        })

    # Print results
    print("\nComparison of Algorithms with Varied User Preferences:")
    print("=" * 80)
    for result in results:
        print(f"\nAlgorithm: {result['Algorithm']}")
        print(f"Source-Target: {result['Source-Target']}")
        print(f"Avg Execution Time: {result['Avg Execution Time']:.4f} seconds")
        print(f"Std Execution Time: {result['Std Execution Time']:.4f} seconds")
        print(f"Avg Path Cost: {result['Avg Path Cost']:.2f}")
        print(f"Avg Path Length: {result['Avg Path Length']:.2f}")
        print(f"Success Rate: {result['Success Rate']:.2%}")
        print(f"Unique Routes: {result['Unique Routes']}")
        print("=" * 80)

    # Plot graphs one by one
    plot_graphs_one_by_one(results)

    # Plot trend graph
    plot_trend_graph(results)

# Function to plot graphs one by one
def plot_graphs_one_by_one(results):
    algorithms = [result['Algorithm'] for result in results]
    avg_execution_times = [result['Avg Execution Time'] for result in results]
    avg_path_costs = [result['Avg Path Cost'] for result in results]
    success_rates = [result['Success Rate'] for result in results]
    unique_routes = [result['Unique Routes'] for result in results]

    # Plot Avg Execution Time
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, avg_execution_times, color='blue')
    plt.title('Avg Execution Time')
    plt.ylabel('Seconds')
    plt.xlabel('Algorithm')
    plt.show()

    # Plot Avg Path Cost
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, avg_path_costs, color='green')
    plt.title('Avg Path Cost')
    plt.ylabel('Cost')
    plt.xlabel('Algorithm')
    plt.show()

    # Plot Success Rate
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, success_rates, color='orange')
    plt.title('Success Rate')
    plt.ylabel('Rate')
    plt.xlabel('Algorithm')
    plt.show()

    # Plot Unique Routes
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, unique_routes, color='red')
    plt.title('Unique Routes')
    plt.ylabel('Count')
    plt.xlabel('Algorithm')
    plt.show()

# Function to plot trend graph
def plot_trend_graph(results):
    algorithms = [result['Algorithm'] for result in results]
    avg_execution_times = [result['Avg Execution Time'] for result in results]
    avg_path_costs = [result['Avg Path Cost'] for result in results]
    success_rates = [result['Success Rate'] for result in results]
    unique_routes = [result['Unique Routes'] for result in results]

    # Plot trend graph
    plt.figure(figsize=(12, 6))

    # Plot Avg Execution Time trend
    plt.subplot(2, 2, 1)
    plt.plot(algorithms, avg_execution_times, marker='o', color='blue', label='Execution Time')
    plt.title('Execution Time Trend')
    plt.ylabel('Seconds')
    plt.xlabel('Algorithm')
    plt.legend()

    # Plot Avg Path Cost trend
    plt.subplot(2, 2, 2)
    plt.plot(algorithms, avg_path_costs, marker='o', color='green', label='Path Cost')
    plt.title('Path Cost Trend')
    plt.ylabel('Cost')
    plt.xlabel('Algorithm')
    plt.legend()

    # Plot Success Rate trend
    plt.subplot(2, 2, 3)
    plt.plot(algorithms, success_rates, marker='o', color='orange', label='Success Rate')
    plt.title('Success Rate Trend')
    plt.ylabel('Rate')
    plt.xlabel('Algorithm')
    plt.legend()

    # Plot Unique Routes trend
    plt.subplot(2, 2, 4)
    plt.plot(algorithms, unique_routes, marker='o', color='red', label='Unique Routes')
    plt.title('Unique Routes Trend')
    plt.ylabel('Count')
    plt.xlabel('Algorithm')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
base_user_prefs = {
    'distance': 1.0,
    'cost': 1.0,
    'scenic': 1.0,
    'comfort': 1.0,
    'disabled_friendly': 0.0,
}

source = "AG10"  # AMPANG
target = "PY04"  # SEMANTAN

compare_algorithms_with_varied_prefs(graph, source, target, base_user_prefs, num_runs=10, num_variations=3, variation_factor=0.1)

