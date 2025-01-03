from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import random
from datetime import datetime
import ast
import os

app = Flask(__name__)

# Load GTFS data with error handling
try:
    stops = pd.read_csv("Original GTFS Data/stops.csv")
    stop_times = pd.read_csv("Original GTFS Data/stop_times.csv")
    routes = pd.read_csv("Original GTFS Data/routes.csv")
    trips = pd.read_csv("Original GTFS Data/trips.csv")
    routes_with_pois = pd.read_csv("GTFS Data Creation/routes_with_pois.csv")
    trips_with_pois = pd.read_csv("GTFS Data Creation/trips_with_pois.csv")
    transfers = pd.read_csv("GTFS Data Creation/transfers.csv")
    print("All data files loaded successfully")
except Exception as e:
    print(f"Error loading data files: {e}")
    raise

# Initialize the graph
graph = nx.Graph()

# Helper function to calculate dynamic fare
def calculate_dynamic_fare_with_increment(distance):
    initial_fare = 0.20
    max_increment = 0.50
    max_distance = 10000
    increment = (max_increment - 0.10) * min(distance, max_distance) / max_distance + 0.10
    return initial_fare + increment

# Helper function to calculate time difference
def calculate_time_difference(time1, time2):
    time_format = "%H:%M:%S"
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)
    return (t2 - t1).total_seconds()

# Add nodes (stops)
for _, stop in stops.iterrows():
    graph.add_node(
        stop['stop_id'],
        stop_name=stop['stop_name'],
        location=(stop['stop_lat'], stop['stop_lon']),
        category=stop.get('category', None),
        isOKU=stop.get('isOKU', None),
        nearby_pois=stop.get('nearby_pois', '[]'),
        search_name=stop.get('search', ''),
        additional_data=stop.to_dict()
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
        previous_stop = None
        previous_departure_time = None
        current_trip_id = trip_id

    if previous_stop is not None:
        route_id = trip_route_map.get(trip_id)
        
        # Calculate distance between stops
        loc1 = (stops.loc[stops['stop_id'] == previous_stop, 'stop_lat'].iloc[0],
                stops.loc[stops['stop_id'] == previous_stop, 'stop_lon'].iloc[0])
        loc2 = (stops.loc[stops['stop_id'] == stop_id, 'stop_lat'].iloc[0],
                stops.loc[stops['stop_id'] == stop_id, 'stop_lon'].iloc[0])
        distance = geodesic(loc1, loc2).meters

        # Calculate time between stops
        time_taken = calculate_time_difference(previous_departure_time, arrival_time)

        # Calculate fare
        fare = calculate_dynamic_fare_with_increment(distance)

        # Add edge if it doesn't exist or update if shorter distance
        if not graph.has_edge(previous_stop, stop_id) or \
           graph[previous_stop][stop_id]['weight'] > distance:
            graph.add_edge(
                previous_stop,
                stop_id,
                weight=distance,
                time_taken=time_taken,
                route_id=route_id,
                fee=fare
            )

    previous_stop = stop_id
    previous_departure_time = departure_time

# Add transfer edges
for _, transfer in transfers.iterrows():
    from_stop = transfer['from_stop_id']
    to_stop = transfer['to_stop_id']
    
    # Calculate transfer distance
    loc1 = (stops.loc[stops['stop_id'] == from_stop, 'stop_lat'].iloc[0],
            stops.loc[stops['stop_id'] == from_stop, 'stop_lon'].iloc[0])
    loc2 = (stops.loc[stops['stop_id'] == to_stop, 'stop_lat'].iloc[0],
            stops.loc[stops['stop_id'] == to_stop, 'stop_lon'].iloc[0])
    distance = geodesic(loc1, loc2).meters

    # Add transfer edge
    graph.add_edge(
        from_stop,
        to_stop,
        weight=distance,
        time_taken=transfer['min_transfer_time'],
        is_transfer=True,
        fee=0.0  # No fee for transfers
    )

def ant_colony_route(graph, source, target, user_prefs, n_ants=20, n_iterations=100, evaporation_rate=0.1):
    if source not in graph or target not in graph:
        return None, float('inf')

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize pheromone on edges with a higher base value
    base_pheromone = 2.0
    for u, v in graph.edges():
        graph[u][v]['pheromone'] = base_pheromone

    best_path = None
    best_path_cost = float('inf')
    
    # Keep track of best paths across iterations
    all_best_paths = []
    all_best_costs = []

    for iteration in range(n_iterations):
        paths = []
        path_costs = []

        for ant in range(n_ants):
            current = source
            path = [current]
            path_cost = 0
            visited = {current}  # Use set for faster lookup

            while current != target:
                neighbors = [n for n in graph.neighbors(current) if n not in visited]
                if not neighbors:
                    break

                # Calculate probabilities for next step
                probabilities = []
                total_attractiveness = 0
                
                for neighbor in neighbors:
                    edge = graph[current][neighbor]
                    distance = edge['weight']
                    time = edge.get('time_taken', 0)
                    fee = edge.get('fee', 0)
                    is_transfer = edge.get('is_transfer', False)
                    
                    # Calculate base cost with user preferences
                    cost = (
                        user_prefs['distance'] * distance +
                        user_prefs['cost'] * fee +
                        user_prefs['comfort'] * (2000 if is_transfer else 0)  # Increased transfer penalty
                    )
                    
                    # Add scenic preference
                    if user_prefs['scenic'] > 0:
                        nearby_pois = graph.nodes[neighbor].get('nearby_pois', [])
                        scenic_value = len(ast.literal_eval(nearby_pois)) if isinstance(nearby_pois, str) else len(nearby_pois)
                        cost -= user_prefs['scenic'] * scenic_value * 200  # Increased scenic bonus

                    # Add disabled-friendly preference
                    if user_prefs['disabled_friendly'] > 0 and not graph.nodes[neighbor].get('isOKU', False):
                        cost += 2000  # Increased penalty for non-OKU stations

                    # Calculate attractiveness with modified formula
                    pheromone = edge['pheromone']
                    attractiveness = (pheromone ** 1.5) * ((1.0 / (cost + 1)) ** 2)  # Modified exponents
                    probabilities.append((neighbor, attractiveness))
                    total_attractiveness += attractiveness

                if not probabilities or total_attractiveness == 0:
                    break

                # Select next step using weighted random choice
                r = random.random() * total_attractiveness
                cum_prob = 0
                
                for neighbor, prob in probabilities:
                    cum_prob += prob
                    if cum_prob >= r:
                        next_stop = neighbor
                        break
                else:
                    next_stop = probabilities[-1][0]

                path.append(next_stop)
                visited.add(next_stop)
                path_cost += graph[current][next_stop]['weight']
                current = next_stop

            if current == target:
                paths.append(path)
                path_costs.append(path_cost)

                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
                    all_best_paths.append(path)
                    all_best_costs.append(path_cost)

        # Update pheromones with increased deposit
        for u, v in graph.edges():
            graph[u][v]['pheromone'] *= (1 - evaporation_rate)

        for path, cost in zip(paths, path_costs):
            deposit = 2.0 / cost  # Increased pheromone deposit
            for u, v in zip(path[:-1], path[1:]):
                graph[u][v]['pheromone'] += deposit

    # Return the most frequent best path
    if all_best_paths:
        # Convert paths to tuples for counting
        path_tuples = [tuple(path) for path in all_best_paths]
        most_common_path = max(set(path_tuples), key=path_tuples.count)
        return list(most_common_path), min(all_best_costs)
    
    return best_path, best_path_cost


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stations')
def get_stations():
    stations = []
    for _, row in stops.iterrows():
        stations.append({
            'label': row['stop_name'],
            'value': row['stop_id'],
            'type': row['category'],
            'route': row['route_id'],
            'isOKU': row['isOKU'],
            'search': row['search']
        })
    return jsonify(stations)

@app.route('/find_routes', methods=['POST'])
def find_routes():
    data = request.get_json()
    
    # Extract preferences
    user_prefs = {
        'distance': float(data['distance']),
        'cost': float(data['cost']),
        'scenic': float(data['scenic']),
        'comfort': float(data['comfort']),
        'disabled_friendly': 1.0 if data['oku'] else 0.0
    }

    try:
        # Get route segments
        route_segments = data['routes']
        all_routes = []

        # Process each segment
        for segment in route_segments:
            source = segment['from']
            target = segment['to']

            # Find best path for this segment
            best_path, best_cost = ant_colony_route(graph, source, target, user_prefs)

            if best_path:
                # Calculate step-by-step details for this segment
                steps = []
                current_route_id = None
                current_segment = None

                for u, v in zip(best_path[:-1], best_path[1:]):
                    edge_data = graph[u][v]
                    is_transfer = edge_data.get('is_transfer', False)
                    route_id = edge_data.get('route_id', 'Unknown')

                    # If it's a transfer or the route changes, finalize the current segment
                    if is_transfer or (current_route_id and route_id != current_route_id):
                        if current_segment:
                            steps.append(current_segment)
                        current_segment = None

                    # If it's a transfer, add it as a separate step
                    if is_transfer:
                        steps.append({
                            'from': graph.nodes[u]['stop_name'],
                            'to': graph.nodes[v]['stop_name'],
                            'distance': f"{edge_data['weight'] / 1000:.2f} km",
                            'time_taken': f"{edge_data['time_taken'] / 60:.0f} mins",
                            'cost': f"RM {edge_data['fee']:.2f}",
                            'route_id': 'Walking',
                            'is_transfer': True
                        })
                    else:
                        # If it's the same route, accumulate the segment
                        if current_segment and route_id == current_route_id:
                            current_segment['to'] = graph.nodes[v]['stop_name']
                            current_segment['distance'] = f"{float(current_segment['distance'].split()[0]) + edge_data['weight'] / 1000:.2f} km"
                            current_segment['time_taken'] = f"{float(current_segment['time_taken'].split()[0]) + edge_data['time_taken'] / 60:.0f} mins"
                            current_segment['cost'] = f"RM {float(current_segment['cost'].replace('RM ', '')) + edge_data['fee']:.2f}"
                        else:
                            # Start a new segment
                            current_segment = {
                                'from': graph.nodes[u]['stop_name'],
                                'to': graph.nodes[v]['stop_name'],
                                'distance': f"{edge_data['weight'] / 1000:.2f} km",
                                'time_taken': f"{edge_data['time_taken'] / 60:.0f} mins",
                                'cost': f"RM {edge_data['fee']:.2f}",
                                'route_id': route_id,
                                'is_transfer': False
                            }
                            current_route_id = route_id

                # Add the last segment if it exists
                if current_segment:
                    steps.append(current_segment)

                # Add the segment details to the route
                route = {
                    'steps': steps,
                    'total_distance': f"{sum(graph[u][v]['weight'] for u, v in zip(best_path[:-1], best_path[1:])) / 1000:.2f} km",
                    'total_time': f"{sum(graph[u][v]['time_taken'] for u, v in zip(best_path[:-1], best_path[1:])) / 60:.0f} mins",
                    'total_cost': f"RM {sum(graph[u][v]['fee'] for u, v in zip(best_path[:-1], best_path[1:])):.2f}",
                    'interchanges': sum(1 for u, v in zip(best_path[:-1], best_path[1:]) if graph[u][v].get('is_transfer', False))
                }
                all_routes.append(route)
            else:
                return jsonify({'error': f'No valid route found for segment {source} to {target}'}), 404

        # Return the step-by-step details for all segments
        return jsonify(all_routes)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)