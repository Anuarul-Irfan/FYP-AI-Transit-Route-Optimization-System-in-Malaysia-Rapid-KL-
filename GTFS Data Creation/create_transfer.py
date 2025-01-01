import pandas as pd

# Load stops.csv
stops_df = pd.read_csv("stops.csv")  # Replace with the actual file path

# Identify valid interchanges
validated_interchanges = []

# Group stops by stop_name and create connections for interchanges
for stop_name, group in stops_df.groupby('stop_name'):
    if len(group) > 1:  # If multiple IDs share the same name
        stop_ids = group['stop_id'].tolist()
        for i in range(len(stop_ids)):
            for j in range(i + 1, len(stop_ids)):
                validated_interchanges.append({
                    "from_stop_id": stop_ids[i],
                    "to_stop_id": stop_ids[j],
                    "transfer_type": 0,
                    "min_transfer_time": 180
                })
transfers = [
    {"from_stop_id": "KG16", "to_stop_id": "KJ14", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "KG22", "to_stop_id": "AG13", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "AG8", "to_stop_id": "KG17", "transfer_type": 0, "min_transfer_time": 120},
    {"from_stop_id": "AG9", "to_stop_id": "MR4", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "PY29", "to_stop_id": "PY28", "transfer_type": 0, "min_transfer_time": 120},
    {"from_stop_id": "PY20", "to_stop_id": "KJ9", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "KG04", "to_stop_id": "PY01", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "KJ31", "to_stop_id": "BRT7", "transfer_type": 0, "min_transfer_time": 120},
    {"from_stop_id": "KJ26", "to_stop_id": "BRT1", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "AG1", "to_stop_id": "SP1", "transfer_type": 0, "min_transfer_time": 120},
    {"from_stop_id": "KG15", "to_stop_id": "KJ15", "transfer_type": 0, "min_transfer_time": 300},
    {"from_stop_id": "MR11", "to_stop_id": "AG3", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "MR9", "to_stop_id": "AG5", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "MR11", "to_stop_id": "PY17", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "MR11", "to_stop_id": "KJ3", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "AG3", "to_stop_id": "PY17", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "AG3", "to_stop_id": "KJ3", "transfer_type": 0, "min_transfer_time": 180},
    {"from_stop_id": "PY17", "to_stop_id": "KJ3", "transfer_type": 0, "min_transfer_time": 180},
]

# Save to CSV
validated_interchanges_df = pd.DataFrame(validated_interchanges + transfers)
validated_interchanges_df.to_csv("validated_interchanges_from_stops.csv", index=False)
print("Validated interchanges saved to validated_interchanges_from_stops.csv")
