import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Load GTFS data
stops = pd.read_csv("stops.csv")
stop_times = pd.read_csv("stop_times.csv")
routes_with_pois = pd.read_csv("routes_with_pois.csv")
trips_with_pois = pd.read_csv("trips_with_pois.csv")

# Initialize the graph
graph = nx.Graph()

# Fee calculation parameters based on distance ranges
fee_ranges = [
    (0.8, 1.3, 0.80),
    (1.3, 1.8, 1.30),
    (1.8, 2.5, 1.80),
    (2.5, 3.0, 2.50),
    (3.0, 3.5, 3.00),
    (3.5, 4.0, 3.50),
    (4.0, 4.5, 4.00),
    (4.5, 5.0, 4.50),
    (5.0, 5.5, 5.00),
    (5.5, 6.0, 5.50),
    (6.0, 7.0, 6.00),
    (7.0, float('inf'), 7.00),
]

def calculate_fee(distance):
    for lower, upper, fee in fee_ranges:
        if lower <= distance < upper:
            return fee
    return 7.00  # Default max fee

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
        additional_data=stop.to_dict()  # Store all stop attributes
    )

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

        # Calculate fee based on distance ranges
        fee = calculate_fee(distance)

        # Calculate time taken from stop_times data
        time_taken = calculate_time_difference(previous_departure_time, arrival_time)

        # Define scenic score
        scenic_score = len(graph.nodes[previous_stop].get('nearby_pois', [])) + len(graph.nodes[stop_id].get('nearby_pois', []))

        graph.add_edge(
            previous_stop,
            stop_id,
            weight=distance,  # Add distance as weight
            pheromone=1.0,    # Initial pheromone level for ACO
            visibility=1.0 / distance if distance > 0 else 0,  # Inverse of distance
            trip_id=trip_id,
            route_id=stop_time.get('route_id', None),
            fee=fee,  # Fee based on distance range
            time_taken=time_taken,  # Time taken in seconds
            scenic_score=scenic_score,  # Add scenic score
            schedule_data=stop_time.to_dict()  # Store all stop_time attributes
        )

    previous_stop = stop_id
    previous_departure_time = departure_time

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

# Plot the graph
plt.figure(figsize=(12, 8))
pos = {node: (data['location'][1], data['location'][0]) for node, data in graph.nodes(data=True)}
nx.draw(
    graph,
    pos,
    node_size=50,
    node_color="blue",
    edge_color="gray",
    with_labels=True,
    labels={node: data['stop_name'] for node, data in graph.nodes(data=True)}
)

# Annotate edges with relevant information
edge_labels = {
    (u, v): f"Distance: {data['weight']:.2f}m\nFee: RM{data['fee']:.2f}\nScenic: {data['scenic_score']}\nTime: {data['time_taken']:.2f}s"
    for u, v, data in graph.edges(data=True)
}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

plt.title("Transit Network with Nodes and Edges Annotated", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.show()

# Save the graph to a file
with open("gtfs_graph_enriched.gpickle", "wb") as f:
    pickle.dump(graph, f)

print("Enriched graph created, plotted, and saved as gtfs_graph_enriched.gpickle.")
