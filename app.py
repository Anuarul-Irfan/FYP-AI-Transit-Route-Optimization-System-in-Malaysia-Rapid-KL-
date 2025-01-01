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

def ant_colony_route(graph, source, target, user_prefs, n_ants=10, n_iterations=50, evaporation_rate=0.1):
    if source not in graph or target not in graph:
        return None, float('inf')

    # Initialize pheromone on edges
    for u, v in graph.edges():
        graph[u][v]['pheromone'] = 1.0

    best_path = None
    best_path_cost = float('inf')

    for iteration in range(n_iterations):
        paths = []
        path_costs = []

        for ant in range(n_ants):
            current = source
            path = [current]
            path_cost = 0

            while current != target:
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break

                # Calculate probabilities for next step
                probabilities = []
                for neighbor in neighbors:
                    if neighbor not in path:
                        edge = graph[current][neighbor]
                        distance = edge['weight']
                        time = edge.get('time_taken', 0)
                        fee = edge.get('fee', 0)
                        is_transfer = edge.get('is_transfer', False)
                        
                        # Factor in user preferences
                        cost = (
                            user_prefs['distance'] * distance +
                            user_prefs['cost'] * fee +
                            user_prefs['comfort'] * (1000 if is_transfer else 0)
                        )
                        
                        # Add scenic preference
                        if user_prefs['scenic'] > 0:
                            nearby_pois = graph.nodes[neighbor].get('nearby_pois', [])
                            scenic_value = len(ast.literal_eval(nearby_pois)) if isinstance(nearby_pois, str) else len(nearby_pois)
                            cost -= user_prefs['scenic'] * scenic_value * 100

                        # Add disabled-friendly preference
                        if user_prefs['disabled_friendly'] > 0:
                            if not graph.nodes[neighbor].get('isOKU', False):
                                cost += 1000

                        pheromone = edge['pheromone']
                        probability = (pheromone ** 2) * ((1.0 / cost) ** 3)
                        probabilities.append((neighbor, probability))

                if not probabilities:
                    break

                # Select next step
                total = sum(p[1] for p in probabilities)
                if total == 0:
                    break

                r = random.random() * total
                cum_prob = 0
                next_stop = None

                for neighbor, prob in probabilities:
                    cum_prob += prob
                    if cum_prob >= r:
                        next_stop = neighbor
                        break

                if next_stop is None:
                    break

                path.append(next_stop)
                path_cost += graph[current][next_stop]['weight']
                current = next_stop

            if current == target:
                paths.append(path)
                path_costs.append(path_cost)

                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost

        # Update pheromones
        for u, v in graph.edges():
            graph[u][v]['pheromone'] *= (1 - evaporation_rate)

        for path, cost in zip(paths, path_costs):
            for u, v in zip(path[:-1], path[1:]):
                graph[u][v]['pheromone'] += 1.0 / cost

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
                # Calculate route details for this segment
                total_distance = sum(graph[u][v]['weight'] for u, v in zip(best_path[:-1], best_path[1:])) / 1000
                total_time = sum(graph[u][v].get('time_taken', 0) for u, v in zip(best_path[:-1], best_path[1:])) / 60
                total_cost = sum(graph[u][v].get('fee', 0.0) for u, v in zip(best_path[:-1], best_path[1:]))
                interchanges = sum(1 for u, v in zip(best_path[:-1], best_path[1:]) if graph[u][v].get('is_transfer', False))
                
                # Handle scenic spots
                scenic_spots = []
                for stop in best_path:
                    pois = graph.nodes[stop].get('nearby_pois', '[]')
                    if isinstance(pois, str):
                        try:
                            pois = ast.literal_eval(pois)
                        except:
                            pois = []
                    scenic_spots.extend([spot for spot, _ in pois])

                # Create step-by-step instructions
                steps = []
                current_line = None

                for from_stop, to_stop in zip(best_path[:-1], best_path[1:]):
                    edge_data = graph[from_stop][to_stop]
                    
                    if edge_data.get('is_transfer', False):
                        steps.append({
                            'type': 'walk',
                            'duration': f"{edge_data.get('time_taken', 3)} mins",
                            'from': graph.nodes[from_stop]['search_name'],
                            'to': graph.nodes[to_stop]['search_name']
                        })
                    else:
                        line = edge_data.get('route_id')
                        if line != current_line:
                            current_step = {
                                'type': stops[stops['stop_id'] == from_stop]['category'].iloc[0],
                                'line': line,
                                'from': graph.nodes[from_stop]['search_name'],
                                'to': graph.nodes[to_stop]['search_name']
                            }
                            steps.append(current_step)
                            current_line = line
                        else:
                            current_step['to'] = graph.nodes[to_stop]['search_name']
                
                route = {
                    'duration': f"{total_time:.0f} mins",
                    'cost': f"RM {total_cost:.2f}",
                    'distance': f"{total_distance:.2f} km",
                    'interchanges': interchanges,
                    'scenic_spots': list(set(scenic_spots)),
                    'oku_friendly': all(stops[stops['stop_id'] == stop]['isOKU'].iloc[0] for stop in best_path),
                    'steps': steps
                }
                
                all_routes.append(route)
            else:
                return jsonify({'error': f'No valid route found for segment {source} to {target}'}), 404

        # Combine all segment routes into one response
        combined_route = {
            'duration': f"{sum(float(r['duration'].split()[0]) for r in all_routes):.0f} mins",
            'cost': f"RM {sum(float(r['cost'].replace('RM ', '')) for r in all_routes):.2f}",
            'distance': f"{sum(float(r['distance'].split()[0]) for r in all_routes):.2f} km",
            'interchanges': sum(r['interchanges'] for r in all_routes),
            'scenic_spots': list(set([spot for r in all_routes for spot in r['scenic_spots']])),
            'oku_friendly': all(r['oku_friendly'] for r in all_routes),
            'steps': [step for r in all_routes for step in r['steps']]
        }
        
        return jsonify([combined_route])
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 