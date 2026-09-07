# src/preprocessing/sensor_groups.py

def build_sensor_groups(config):
    groups = []
    columns_cfg = config["columns"]

    for sensor_name, modalities in columns_cfg.items():
        if sensor_name in ("timestamp", "activity_id"):
            continue
        if not isinstance(modalities, dict):
            continue

        for modality, cols in modalities.items():
            if not isinstance(cols, list) or len(cols) != 3:
                continue
            if modality == 'temperature':
                continue
            axes = tuple(cols)
            type_name = axes[0].rsplit("_", 1)[0]   # same logic as in get_filtered_feature_names
            groups.append({
                "axes": axes,
                "is_accel": "accel" in modality.lower(),
                "type": type_name,
            })

    return groups


def get_filtered_feature_names(sensor_groups):
    names = []
    for group in sensor_groups:
        type_name = group["type"]
        for axis in group["axes"]:
            names.append(f"{axis}_filtered")
        names.append(f"magnitude_{type_name}")
        names.append(f"magnitude_{type_name}_filtered")
    return names


def get_all_feature_names(config, sensor_groups, include_raw=True, include_temperature=True):
    names = []


    if include_raw:
        for group in sensor_groups:
            names.extend(group["axes"])

    names.extend(get_filtered_feature_names(sensor_groups))

    if include_temperature:
        columns_cfg = config["columns"]
        for sensor_name, modalities in columns_cfg.items():
            if sensor_name in ("timestamp", "activity_id"):
                continue
            if not isinstance(modalities, dict):
                continue
            temp_col = modalities.get("temperature")
            if isinstance(temp_col, str):
                names.append(temp_col)

    return names