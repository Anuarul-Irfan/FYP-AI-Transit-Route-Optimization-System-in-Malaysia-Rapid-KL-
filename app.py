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
                        user_prefs['distance'] * distance +  # Higher distance preference = prioritize shorter distance
                        user_prefs['cost'] * fee +           # Higher cost preference = prioritize lower cost
                        user_prefs['comfort'] * (5000 if is_transfer else 0)  # Higher comfort preference = prioritize fewer transfers
                    )
                    
                    # Add scenic preference
                    if user_prefs['scenic'] > 0:
                        nearby_pois = graph.nodes[neighbor].get('nearby_pois', [])
                        scenic_value = len(ast.literal_eval(nearby_pois)) if isinstance(nearby_pois, str) else len(nearby_pois)
                        cost -= user_prefs['scenic'] * scenic_value * 200  # Higher scenic preference = prioritize more scenic points
                        # Subtracting scenic_value from cost because higher scenic_value should reduce the overall cost

                    # Add disabled-friendly preference
                    if user_prefs['disabled_friendly'] > 0 and not graph.nodes[neighbor].get('isOKU', False):
                        cost += 2000  # Penalty for non-OKU stations

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

def generate_multiple_routes(graph, source, target, user_prefs, n_routes=4):
    routes = []
    unique_paths = set()  # To track unique paths

    # Generate routes with slightly modified preferences
    for i in range(n_routes):
        modified_prefs = user_prefs.copy()

        # Slightly tweak the preferences for each route
        if i == 0:
            # Route 1: User's original preferences
            label = "Your Preferences"
        elif i == 1:
            # Route 2: Slightly prioritize distance
            modified_prefs['distance'] *= 1.2  # Increase distance weight by 20%
            modified_prefs['cost'] *= 0.8      # Decrease cost weight by 20%
            label = "Slightly Prioritize Distance"
        elif i == 2:
            # Route 3: Slightly prioritize cost
            modified_prefs['cost'] *= 1.2      # Increase cost weight by 20%
            modified_prefs['distance'] *= 0.8  # Decrease distance weight by 20%
            label = "Slightly Prioritize Cost"
        elif i == 3:
            # Route 4: Slightly prioritize comfort (fewer transfers)
            modified_prefs['comfort'] *= 1.2   # Increase comfort weight by 20%
            modified_prefs['scenic'] *= 0.8    # Decrease scenic weight by 20%
            label = "Slightly Prioritize Comfort"

        # Find the best path with modified preferences
        best_path, best_cost = ant_colony_route(graph, source, target, modified_prefs)
        
        # Ensure the path is unique
        if best_path and tuple(best_path) not in unique_paths:
            unique_paths.add(tuple(best_path))  # Add the path to the set of unique paths
            routes.append({
                'path': best_path,
                'cost': best_cost,
                'label': label  # Add a label for the route
            })
    
    return routes

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

# ... (keep all imports and initial setup the same) ...

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
        all_segment_routes = []

        # Process each segment separately
        for segment in route_segments:
            # Generate routes for this segment
            routes = generate_multiple_routes(graph, segment['from'], segment['to'], user_prefs, n_routes=4)
            if not routes:
                return jsonify({'error': f'No valid route found for segment {segment["from"]} to {segment["to"]}'}), 404
            all_segment_routes.append(routes)

        # Create multiple combined routes
        combined_routes = []
        
        # For simplicity, we'll use the first 4 combinations
        for route_index in range(min(4, len(all_segment_routes[0]))):
            total_steps = []
            total_distance = 0
            total_time = 0
            total_cost = 0
            total_transfers = 0
            
            # Get the route_index-th route from each segment
            for segment_routes in all_segment_routes:
                route = segment_routes[min(route_index, len(segment_routes)-1)]
                steps = []
                current_route_id = None
                current_segment = None

                for u, v in zip(route['path'][:-1], route['path'][1:]):
                    edge_data = graph[u][v]
                    is_transfer = edge_data.get('is_transfer', False)
                    route_id = edge_data.get('route_id', 'Unknown')

                    # Get POIs for the destination station
                    dest_pois = graph.nodes[v].get('nearby_pois', '[]')
                    if isinstance(dest_pois, str):
                        try:
                            dest_pois = ast.literal_eval(dest_pois)
                        except:
                            dest_pois = []

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
                            'is_transfer': True,
                            'nearby_pois': dest_pois  # Add POIs for transfer destination
                        })
                    else:
                        # If it's the same route, accumulate the segment
                        if current_segment and route_id == current_route_id:
                            current_segment['to'] = graph.nodes[v]['stop_name']
                            current_segment['distance'] = f"{float(current_segment['distance'].split()[0]) + edge_data['weight'] / 1000:.2f} km"
                            current_segment['time_taken'] = f"{float(current_segment['time_taken'].split()[0]) + edge_data['time_taken'] / 60:.0f} mins"
                            current_segment['cost'] = f"RM {float(current_segment['cost'].replace('RM ', '')) + edge_data['fee']:.2f}"
                            current_segment['nearby_pois'] = dest_pois  # Update POIs for destination
                        else:
                            # Start a new segment
                            current_segment = {
                                'from': graph.nodes[u]['stop_name'],
                                'to': graph.nodes[v]['stop_name'],
                                'distance': f"{edge_data['weight'] / 1000:.2f} km",
                                'time_taken': f"{edge_data['time_taken'] / 60:.0f} mins",
                                'cost': f"RM {edge_data['fee']:.2f}",
                                'route_id': route_id,
                                'is_transfer': False,
                                'nearby_pois': dest_pois  # Add POIs for new segment
                            }
                            current_route_id = route_id

                # Add the last segment if it exists
                if current_segment:
                    steps.append(current_segment)

                # Add steps to total and update totals
                total_steps.extend(steps)
                total_distance += sum(graph[u][v]['weight'] for u, v in zip(route['path'][:-1], route['path'][1:])) / 1000
                total_time += sum(graph[u][v]['time_taken'] for u, v in zip(route['path'][:-1], route['path'][1:])) / 60
                total_cost += sum(graph[u][v]['fee'] for u, v in zip(route['path'][:-1], route['path'][1:]))
                total_transfers += sum(1 for step in steps if step.get('is_transfer', False))

            # Add this combined route to the list
            route_label = "Your Preferences" if route_index == 0 else \
                         "Slightly Prioritize Distance" if route_index == 1 else \
                         "Slightly Prioritize Cost" if route_index == 2 else \
                         "Slightly Prioritize Comfort"
            
            combined_routes.append({
                'steps': total_steps,
                'total_distance': f"{total_distance:.2f} km",
                'total_time': f"{total_time:.0f} mins",
                'total_cost': f"RM {total_cost:.2f}",
                'interchanges': total_transfers,
                'label': route_label
            })

        # Return all routes
        return jsonify(combined_routes)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)