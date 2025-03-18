import networkx as nx
import numpy as np
import os
import cv2
from Analysis_FrameExtraction import *

def polylines_to_graph(polylines):
    """Converts merged polylines into a NetworkX graph representation."""
    G = nx.Graph()

    for polyline in polylines:
        for i in range(len(polyline) - 1):
            p1 = tuple(polyline[i])
            p2 = tuple(polyline[i + 1])
            G.add_edge(p1, p2, weight=np.linalg.norm(np.array(p1) - np.array(p2)))  # Edge with distance weight

    return G


def score_vessel_structure(polylines):
    """Scores an image based on the clarity of vessel structure."""
    G = polylines_to_graph(polylines)

    # Total path length (sum of all polyline edges)
    total_length = sum(nx.get_edge_attributes(G, 'weight').values())

    # Count loops (cycles in the graph)
    loop_count = len(list(nx.cycle_basis(G)))  # Extracts independent loops

    # Count branch points (Nodes with degree > 2)
    branch_points = sum(1 for node in G.nodes if G.degree(node) > 2)

    # Compute vessel clarity score
    score = total_length - (5 * loop_count) - (2 * branch_points)

    return score


def select_best_vessel_image(image_folder):
    """
    Selects the best vessel image based on polyline structures.
    - image_folder: Path containing polyline files.
    Returns: Best image filename
    """
    best_score = -np.inf
    best_image = None

    for filename in os.listdir(image_folder):
        if filename.endswith("_final_polylines.jpg"):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Assume polylines are extracted using previous processing pipeline
            # polylines = extract_polylines_from_image(img)  # Placeholder for actual function
            score = score_vessel_structure(polylines)
            print(f"Image: {filename}, Score: {score}")

            if score > best_score:
                best_score = score
                best_image = filename

    return best_image


# Example Usage
image_folder = "D:/Cardio/Videos/Media1out"
best_image = select_best_vessel_image(image_folder)
print(f"Best Vessel Image: {best_image}")
