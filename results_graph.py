import matplotlib.pyplot as plt
import numpy as np

# Data from the comparison
algorithms = ["A*", "Ant Colony Optimization", "Dijkstra", "Genetic Algorithm"]
avg_execution_times = [0.0135, 0.1611, 0.0000, 0.0017]
avg_path_costs = [19876.50, 81945.58, 0.00, float('inf')]
success_rates = [1.00, 1.00, 0.00, 1.00]
unique_routes = [1, 2, 0, 1]

# Replace 'inf' with a large number for visualization purposes
avg_path_costs = [cost if cost != float('inf') else 100000 for cost in avg_path_costs]

# Plot Avg Execution Time
plt.figure(figsize=(10, 6))
plt.bar(algorithms, avg_execution_times, color='blue')
plt.title('Average Execution Time')
plt.ylabel('Seconds')
plt.xlabel('Algorithm')
plt.show()

# Plot Avg Path Cost
plt.figure(figsize=(10, 6))
plt.bar(algorithms, avg_path_costs, color='green')
plt.title('Average Path Cost')
plt.ylabel('Cost')
plt.xlabel('Algorithm')
plt.show()

# Plot Success Rate
plt.figure(figsize=(10, 6))
plt.bar(algorithms, success_rates, color='orange')
plt.title('Success Rate')
plt.ylabel('Rate')
plt.xlabel('Algorithm')
plt.ylim(0, 1.1)
plt.show()

# Plot Unique Routes
plt.figure(figsize=(10, 6))
plt.bar(algorithms, unique_routes, color='red')
plt.title('Unique Routes')
plt.ylabel('Count')
plt.xlabel('Algorithm')
plt.show()

# Plot all metrics in a single figure for comparison
plt.figure(figsize=(14, 8))

# Plot Avg Execution Time
plt.subplot(2, 2, 1)
plt.bar(algorithms, avg_execution_times, color='blue')
plt.title('Average Execution Time')
plt.ylabel('Seconds')
plt.xlabel('Algorithm')

# Plot Avg Path Cost
plt.subplot(2, 2, 2)
plt.bar(algorithms, avg_path_costs, color='green')
plt.title('Average Path Cost')
plt.ylabel('Cost')
plt.xlabel('Algorithm')

# Plot Success Rate
plt.subplot(2, 2, 3)
plt.bar(algorithms, success_rates, color='orange')
plt.title('Success Rate')
plt.ylabel('Rate')
plt.xlabel('Algorithm')
plt.ylim(0, 1.1)

# Plot Unique Routes
plt.subplot(2, 2, 4)
plt.bar(algorithms, unique_routes, color='red')
plt.title('Unique Routes')
plt.ylabel('Count')
plt.xlabel('Algorithm')

plt.tight_layout()
plt.show()