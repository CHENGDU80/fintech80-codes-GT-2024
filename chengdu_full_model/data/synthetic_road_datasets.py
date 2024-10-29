import random
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from collections import defaultdict
import re

np.random.seed(12)


def create_synth_data(G, orig_node, dest_node, bbox=(30.5493864, 104.057658, 30.6724752, 104.1682525)):

    # ----------------- Crash Data Generation -----------------
    # Generate random crash points within the bounding box
    num_random_crashes = 200
    random_crash_lats = np.random.uniform(bbox[0], bbox[2], num_random_crashes)
    random_crash_lons = np.random.uniform(bbox[1], bbox[3], num_random_crashes)
    random_crash_points = list(zip(random_crash_lats, random_crash_lons))

    # Generate clusters of crash points
    cluster_centers = [
        (30.6, 104.1),
        (30.62, 104.12),
        (30.65, 104.15)
    ]
    points_per_cluster = 50
    std_dev = 0.008  # Standard deviation for clusters (~200 meters)
    cluster_crash_points = []

    for center_lat, center_lon in cluster_centers:
        cluster_lats = np.random.normal(
            center_lat, std_dev, points_per_cluster)
        cluster_lons = np.random.normal(
            center_lon, std_dev, points_per_cluster)
        cluster_points = list(zip(cluster_lats, cluster_lons))
        cluster_crash_points.extend(cluster_points)

    # Combine random crashes and cluster crashes
    crash_points = random_crash_points + cluster_crash_points

    # Snap crash points to the nearest edges
    crash_points_snapped = []
    for lat, lon in crash_points:
        nearest_edge = ox.nearest_edges(G, lon, lat)
        u, v, key = nearest_edge
        u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
        v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
        edge_line = LineString([(u_x, u_y), (v_x, v_y)])
        point = Point(lon, lat)
        snapped_point = edge_line.interpolate(edge_line.project(point))
        crash_points_snapped.append((snapped_point.x, snapped_point.y))

    # Initialize dictionaries to store point and edge incidents
    point_incidents = {}
    edge_incidents = {}

    # Define the incident types, counts, and whether they are point or edge incidents
    incidents_info = {
        'Disengagement Zones': {'count': 50, 'type': 'point'},
        'High Traffic Density': {'count': 50, 'type': 'edge'},
        'Law Enforcement Activity': {'count': 15, 'type': 'point'},
        'Adverse Weather Conditions': {'count': 20, 'type': 'point'},
        'Crime-Risk Zones': {'count': 20, 'type': 'point'},
        'Pedestrian-Dense Areas': {'count': 50, 'type': 'point'},
        'Poor Road Infrastructure': {'count': 25, 'type': 'edge'},
        'Unplanned Road Work': {'count': 15, 'type': 'edge'},
        'Low Visibility Areas': {'count': 20, 'type': 'point'},
        'Narrow Roads': {'count': 25, 'type': 'edge'}
    }

    # Functions for generating incidents

    def generate_incidents_in_areas(area_polygons, count):
        incidents = []
        for _ in range(count):
            polygon = area_polygons[np.random.randint(0, len(area_polygons))]
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)
                if polygon.contains(point):
                    incidents.append((x, y))
                    break
        return incidents

    def generate_incidents_near_intersections(G, count):
        intersections = [node for node, degree in G.degree() if degree >= 4]
        incidents = []
        for _ in range(count):
            node = np.random.choice(intersections)
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            incidents.append((x, y))
        return incidents

    def generate_incidents_randomly(G, count):
        incidents = []
        nodes = list(G.nodes)
        for _ in range(count):
            node = np.random.choice(nodes)
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            incidents.append((x, y))
        return incidents

    def edges_in_area(G, area_polygons):
        edges_in_area = []
        for u, v, k, data in G.edges(keys=True, data=True):
            line = LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                               (G.nodes[v]['x'], G.nodes[v]['y'])])
            for polygon in area_polygons:
                if line.intersects(polygon):
                    edges_in_area.append((u, v, k))
                    break
        return edges_in_area

    def select_long_edges(G):
        return [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data['length'] > 500]

    def select_main_edges(G):
        return [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('highway') == 'primary']

    def select_narrow_edges(G):
        narrow_edges = []
        for u, v, k, data in G.edges(keys=True, data=True):
            width = data.get('width', '5')
            try:
                width_value = float(
                    re.match(r'^\D*(\d+\.?\d*)', str(width)).group(1))
            except (AttributeError, ValueError):
                width_value = 5
            if width_value < 5:
                narrow_edges.append((u, v, k))
        return narrow_edges

    def generate_edge_incidents(G, count, edge_selector):
        selected_edges = edge_selector(G)
        incidents = []
        for _ in range(count):
            edge = selected_edges[np.random.randint(0, len(selected_edges))]
            incidents.append(edge)
        return incidents

    # Generate incidents for each type
    for incident_type, info in incidents_info.items():
        count = info['count']
        if info['type'] == 'point':
            # Generate point incidents
            if incident_type == 'Disengagement Zones':
                # Generate random points within the network area
                points = []
                for _ in range(count):
                    lat = np.random.uniform(bbox[0], bbox[2])
                    lon = np.random.uniform(bbox[1], bbox[3])
                    points.append((lon, lat))
            elif incident_type == 'Law Enforcement Activity':
                points = generate_incidents_near_intersections(G, count)
            elif incident_type == 'Adverse Weather Conditions':
                points = generate_incidents_randomly(G, count)
            elif incident_type == 'Crime-Risk Zones':
                points = generate_incidents_in_areas(
                    [Point(104.11, 30.6).buffer(0.005),
                     Point(104.13, 30.62).buffer(0.005)],
                    count
                )
            elif incident_type == 'Pedestrian-Dense Areas':
                area_polygons = [
                    Point(104.09, 30.6).buffer(0.004),
                    Point(104.12, 30.61).buffer(0.004),
                    Point(104.11, 30.63).buffer(0.004),
                    Point(104.14, 30.62).buffer(0.004)
                ]
                points = generate_incidents_in_areas(area_polygons, count)
            elif incident_type == 'Low Visibility Areas':
                points = generate_incidents_in_areas(
                    [Point(104.1, 30.6).buffer(0.006)],
                    count
                )
            else:
                points = generate_incidents_randomly(G, count)
            # Snap points to the nearest edge
            snapped_points = []
            for x, y in points:
                nearest_edge = ox.nearest_edges(G, x, y)
                u, v, key = nearest_edge
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
                edge_line = LineString([(u_x, u_y), (v_x, v_y)])
                point = Point(x, y)
                snapped_point = edge_line.interpolate(edge_line.project(point))
                snapped_points.append((snapped_point.x, snapped_point.y))
            # Store the snapped points
            point_incidents[incident_type] = snapped_points
        elif info['type'] == 'edge':
            # Generate edge incidents
            if incident_type == 'High Traffic Density':
                edges = generate_edge_incidents(G, count, select_long_edges)
            elif incident_type == 'Poor Road Infrastructure':
                edges = generate_edge_incidents(G, count, lambda G: edges_in_area(
                    G, [Point(104.06, 30.65).buffer(0.005)]))
            elif incident_type == 'Unplanned Road Work':
                edges = generate_edge_incidents(G, count, select_main_edges)
            elif incident_type == 'Narrow Roads':
                edges = generate_edge_incidents(G, count, select_narrow_edges)
            else:
                edges = generate_edge_incidents(
                    G, count, lambda G: list(G.edges(keys=True)))
            # Store the edges
            edge_incidents[incident_type] = edges

    # Map incident types to severity levels
    severity_mapping = {
        'Disengagement Zones': 10,
        'High Traffic Density': 20,
        'Law Enforcement Activity': 10,
        'Adverse Weather Conditions': 20,
        'Crime-Risk Zones': 30,
        'Pedestrian-Dense Areas': 20,
        'Poor Road Infrastructure': 30,
        'Unplanned Road Work': 20,
        'Low Visibility Areas': 20,
        'Narrow Roads': 10
    }

    # ----------------- Data Preparation for DataFrame -----------------
    # Edge incident mapping and cumulative severity
    edge_incident_mapping = {}
    edge_severity = defaultdict(int)
    for incident_type, edges in edge_incidents.items():
        severity = severity_mapping[incident_type]
        for edge in edges:
            # Assign incident type based on priority if multiple incidents on the same edge
            if edge not in edge_incident_mapping or severity > severity_mapping[edge_incident_mapping[edge]]:
                edge_incident_mapping[edge] = incident_type
                edge_severity[edge] = severity
    # ----------------- Create DataFrame with Edge Information and Variables -----------------

    # Initialize a list to hold the data
    data_list = []

    # For each edge in G
    for u, v, key, data in G.edges(keys=True, data=True):
        edge = (u, v, key)
        edge_data = {}
        edge_data['u'] = u
        edge_data['v'] = v

        # Assign variables based on incidents and attributes
        # 'disengagement_frequency' based on cumulative severity
        edge_data['disengagement_frequency'] = data.get(
            'cumulative_severity', 0)

        # 'traffic_density'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'High Traffic Density':
            edge_data['traffic_density'] = random.uniform(
                7, 10)  # High traffic density
        else:
            edge_data['traffic_density'] = random.uniform(
                1, 6)  # Low to medium traffic density

        # 'law_enforcement_activity'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Law Enforcement Activity':
            edge_data['law_enforcement_activity'] = random.uniform(7, 10)
        else:
            edge_data['law_enforcement_activity'] = random.uniform(1, 6)

        # 'adverse_weather_conditions'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Adverse Weather Conditions':
            edge_data['adverse_weather_conditions'] = random.uniform(7, 10)
        else:
            edge_data['adverse_weather_conditions'] = random.uniform(1, 6)

        # 'crime_risk_score'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Crime-Risk Zones':
            edge_data['crime_risk_score'] = random.uniform(7, 10)
        else:
            edge_data['crime_risk_score'] = random.uniform(1, 6)

        # 'pedestrian_density'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Pedestrian-Dense Areas':
            edge_data['pedestrian_density'] = random.uniform(7, 10)
        else:
            edge_data['pedestrian_density'] = random.uniform(1, 6)

        # 'infrastructure_quality'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Poor Road Infrastructure':
            edge_data['infrastructure_quality'] = random.uniform(
                1, 4)  # Poor quality
        else:
            edge_data['infrastructure_quality'] = random.uniform(
                7, 10)  # Good quality

        # 'road_work_present'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Unplanned Road Work':
            edge_data['road_work_present'] = 1
        else:
            edge_data['road_work_present'] = 0

        # 'visibility_score'
        if edge in edge_incident_mapping and edge_incident_mapping[edge] == 'Low Visibility Areas':
            edge_data['visibility_score'] = random.uniform(
                1, 4)  # Low visibility
        else:
            edge_data['visibility_score'] = random.uniform(
                7, 10)  # Good visibility

        # 'terrain_steepness'
        # For simplicity, let's assign random terrain steepness
        edge_data['terrain_steepness'] = np.random.uniform(0, 15)

        # 'road_width'
        width = data.get('width', '5')
        try:
            width_value = float(
                re.match(r'^\D*(\d+\.?\d*)', str(width)).group(1))
        except (AttributeError, ValueError):
            width_value = random.uniform(5, 10)
        edge_data['road_width'] = width_value

        # Append the edge_data to the list
        data_list.append(edge_data)

    # Create the DataFrame
    sample_data = pd.DataFrame(data_list)

    # ----------------- Generate Wait Times and Severities -----------------

    # Generate wait_times using the same formula as in code b
    n_samples = len(sample_data)
    base_wait = 100
    wait_times = (
        base_wait +
        -5 * sample_data['disengagement_frequency'] +
        -0.3 * sample_data['traffic_density'] +
        2 * sample_data['law_enforcement_activity'] +
        -10 * sample_data['adverse_weather_conditions'] +
        -2 * sample_data['crime_risk_score'] +
        -0.5 * sample_data['pedestrian_density'] +
        3 * sample_data['infrastructure_quality'] +
        -15 * sample_data['road_work_present'] +
        2 * sample_data['visibility_score'] +
        -1 * sample_data['terrain_steepness'] +
        0.5 * sample_data['road_width'] +
        np.random.normal(0, 5, n_samples)
    )
    wait_times = np.maximum(wait_times, 1)

    # Generate severities
    severities = np.random.choice(
        [1, 2, 3, 4], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

    return sample_data, wait_times, severities
