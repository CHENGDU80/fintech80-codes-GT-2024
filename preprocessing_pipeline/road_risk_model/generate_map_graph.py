import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import networkx as nx
import osmnx as ox
import re
from collections import defaultdict
import random
from typing import Dict, Tuple, Optional, List, Union
import warnings
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def generate_map_graph(
        bbox=(30.5493864, 104.057658, 30.6724752, 104.1682525)):
    """
    Generates a street network graph within a specified bounding box using OpenStreetMap data.

    Args:
        bbox (tuple): The bounding box defined by (south, west, north, east) coordinates.

    Returns:
        networkx.MultiDiGraph: A directed graph representing the road network within the bounding box.
    """
    # Load the street network for driving within the specified bounding box
    G = ox.graph_from_bbox(
        north=bbox[2], south=bbox[0], east=bbox[3], west=bbox[1], network_type='drive'
    )
    return G


def find_shortest_route(
        G,
        orig_point=(30.5593864, 104.067658),
        dest_point=(30.6624752, 104.1582525)):
    """
    Finds the shortest route between two geographical points in the given street network graph.

    Args:
        G (networkx.MultiDiGraph): The road network graph.
        orig_point (tuple): The latitude and longitude of the origin point.
        dest_point (tuple): The latitude and longitude of the destination point.

    Returns:
        tuple: A tuple containing:
            - shortest_route (list): A list of nodes representing the shortest path from origin to destination.
            - orig_node (int): The node in the graph closest to the origin point.
            - dest_node (int): The node in the graph closest to the destination point.
    """
    # Find the nearest graph nodes to the specified origin and destination points
    orig_node = ox.nearest_nodes(G, orig_point[1], orig_point[0])
    dest_node = ox.nearest_nodes(G, dest_point[1], dest_point[0])

    # Compute the shortest path based on edge length between the nodes
    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')

    return shortest_route, orig_node, dest_node
