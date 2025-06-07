import cv2
import numpy as np
import os
import random
import networkx as nx
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from Analysis_NetworkAlignment import *

def simplified_skeletonize(img):
    """Extracts a simplified skeleton using controlled erosion and thinning."""
    # Apply stronger Gaussian Blur to smooth structures
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    #
    # # Apply morphological opening to remove fine details
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Use cv2 thinning method if available
    if hasattr(cv2.ximgproc, 'thinning'):
        skeleton = cv2.ximgproc.thinning(img)
    else:
        # Controlled skeletonization using erosion
        size = np.array(img.shape)
        skeleton = np.zeros(size, np.uint8)
        eroded = np.copy(img)
        temp = np.zeros(size, np.uint8)

        while cv2.countNonZero(eroded) > 0:
            eroded = cv2.erode(eroded, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)

    return skeleton

def convert_skeleton_to_polylines(skeleton, min_branch_length=0):
    """Converts a cleaned skeleton (2D array) directly into vector polylines, removing short branches."""
    height, width = skeleton.shape
    visited = np.zeros((height, width), dtype=bool)
    polylines = []

    # Define movement directions (8-connectivity)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def count_neighbors(x, y):
        """Counts the number of foreground neighbors of a given pixel."""
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and skeleton[nx, ny] > 0:
                count += 1
        return count

    def trace_polyline(start):
        """Traces a single connected polyline from a starting pixel, ensuring uniqueness."""
        path = []
        stack = [start]

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            path.append((y, x))  # Store as (col, row) for OpenCV format

            # Explore neighbors
            next_steps = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and skeleton[nx, ny] > 0 and not visited[nx, ny]:
                    next_steps.append((nx, ny))

            if len(next_steps) == 1:
                stack.append(next_steps[0])

        return np.array(path, dtype=np.int32) if len(path) > min_branch_length else None

    # Remove short branches before tracing
    for i in range(height):
        for j in range(width):
            if skeleton[i, j] > 0 and count_neighbors(i, j) <= 1:
                skeleton[i, j] = 0  # Remove short dangles

    # Iterate through the cleaned skeleton
    for i in range(height):
        for j in range(width):
            if skeleton[i, j] > 0 and not visited[i, j]:
                polyline = trace_polyline(start=(i, j))
                if polyline is not None:
                    polylines.append(polyline)

    return polylines


def merge_nearby_polylines(polylines, merge_threshold=10):
    """
    Merges polylines that have endpoints close to each other.
    - merge_threshold: Maximum Euclidean distance to merge polylines.
    """
    merged_polylines = []
    used = set()  # Track merged polylines

    def distance(pt1, pt2):
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    for i, poly1 in enumerate(polylines):
        if i in used:
            continue
        merged_poly = poly1.tolist()
        used.add(i)

        for j, poly2 in enumerate(polylines):
            if j in used or i == j:
                continue

            # Check if poly1's end is close to poly2's start
            if distance(merged_poly[-1], poly2[0]) < merge_threshold:
                merged_poly.extend(poly2.tolist())  # Merge them
                used.add(j)

            # Check if poly1's start is close to poly2's end (reverse needed)
            elif distance(merged_poly[0], poly2[-1]) < merge_threshold:
                merged_poly = poly2.tolist() + merged_poly  # Merge in reverse order
                used.add(j)

        merged_polylines.append(np.array(merged_poly, dtype=np.int32))

    return merged_polylines


def visualize_polylines(image, polylines):
    """Draws polylines with different colors."""
    img_with_polylines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for polyline in polylines:
        if len(polyline) < 2:
            continue  # Ignore very short segments

        # Assign a unique random color
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        cv2.polylines(img_with_polylines, [polyline], isClosed=False, color=color, thickness=1)

    return img_with_polylines


def skeleton_to_network(skeleton):
    """Converts the skeleton image into a network graph structure, identifying branches and loops."""
    height, width = skeleton.shape
    visited = np.zeros((height, width), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connectivity

    # Network structure representation
    graph = defaultdict(list)
    node_coords = {}  # Node ID to coordinate mapping
    node_id_counter = 0
    edge_paths = {}  # Edge to path mapping
    edge_lengths = {}  # Stores edge lengths {(node1, node2): length}

    def is_junction_or_endpoint(x, y):
        """Determines whether it is a branch point or endpoint."""
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and skeleton[nx, ny] > 0:
                count += 1
        return count != 2  # It is a branch point or endpoint when the neighbor count ≠ 2

    def trace_branch(start_node, start_x, start_y, initial_dx, initial_dy):
        """Traces a branch from a node and calculates its length."""
        path = [(start_x, start_y)]
        x, y = start_x + initial_dx, start_y + initial_dy
        prev_x, prev_y = start_x, start_y
        length = 0  # Initialize length counter

        while True:
            path.append((x, y))
            visited[x, y] = True

            # Calculate the contribution of the current step (considering diagonal steps)
            dx, dy = x - prev_x, y - prev_y
            length += np.sqrt(dx ** 2 + dy ** 2)  # Euclidean distance

            # Find the next point
            next_points = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < height and 0 <= ny < width and
                        skeleton[nx, ny] > 0 and
                        not (nx == prev_x and ny == prev_y)):
                    next_points.append((nx, ny))

            # Reach another node or endpoint
            if len(next_points) != 1 or is_junction_or_endpoint(x, y):
                end_node = None
                # Check if it reaches a known node
                for node_id, (node_x, node_y) in node_coords.items():
                    if x == node_x and y == node_y:
                        end_node = node_id
                        break

                if end_node is None and is_junction_or_endpoint(x, y):
                    end_node = node_id_counter
                    node_coords[end_node] = (x, y)
                    node_id_counter += 1

                return end_node, path, length

            prev_x, prev_y = x, y
            x, y = next_points[0]

    # Step 1: Identify all nodes (branch points and endpoints)
    nodes = []
    for x in range(height):
        for y in range(width):
            if skeleton[x, y] > 0 and is_junction_or_endpoint(x, y):
                node_id = node_id_counter
                node_coords[node_id] = (x, y)
                nodes.append((x, y, node_id))
                node_id_counter += 1

    # Step 2: Trace all branches
    for node_id, (x, y) in node_coords.items():
        # Check all possible branch directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < height and 0 <= ny < width and
                    skeleton[nx, ny] > 0 and not visited[nx, ny]):

                end_node, path, length = trace_branch(node_id, x, y, dx, dy)
                if end_node is not None:
                    # Add the edge to the graph
                    graph[node_id].append(end_node)
                    graph[end_node].append(node_id)
                    edge_paths[(node_id, end_node)] = path
                    edge_paths[(end_node, node_id)] = path[::-1]
                    edge_lengths[(node_id, end_node)] = length
                    edge_lengths[(end_node, node_id)] = length

    # Calculate the number of loops (using Euler's formula)
    num_nodes = len(node_coords)
    num_edges = sum(len(edges) for edges in graph.values()) // 2
    num_components = 1  # Assume a single connected component (can be calculated via DFS)
    num_loops = num_edges - num_nodes + num_components  # Euler's formula: loop count = edges - nodes + components

    return {
        'graph': graph,
        'node_coords': node_coords,
        'edge_paths': edge_paths,
        'edge_lengths': edge_lengths,
        'num_branches': num_edges,
        'num_loops': num_loops,
    }



if __name__=='__main__':
    # Folder paths (modify as needed)
    input_folder = r".\heart_cycles\Media1"
    output_folder = r".\heart_cycles\Media1_Score"
    os.makedirs(output_folder, exist_ok=True)

    # Define ROI coordinates (top-left and bottom-right)
    roi_top_left = (110, 100)
    roi_bottom_right = (640, 500)

    close_gap_threshold = 10  # Maximum distance to close gaps between polylines
    best_score = -np.inf
    best_image = None
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            base_name = os.path.splitext(filename)[0]  # Extract filename without extension

            # Convert to grayscale and apply ROI
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

            # Apply Gaussian Blur (5x5) to remove small details
            blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            # **Enhanced Thresholding: CLAHE + Otsu's**
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16, 16))
            enhanced = clahe.apply(blurred)

            _, bright_regions = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological closing to merge fragmented strokes
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)

            # Extract simplified skeletonized centerline
            skeleton = simplified_skeletonize(closed)

            # Convert skeleton to a cleaned pixel map
            # pixel_map = convert_skeleton_to_pixel_map(skeleton)

            # Extract polylines from the cleaned pixel map
            polylines = convert_skeleton_to_polylines(skeleton)
            polylines = merge_nearby_polylines(polylines, merge_threshold=15)  # Adjust threshold as needed

            network = skeleton_to_network(skeleton)

            # Calculate total network length
            total_length = sum(network['edge_lengths'].values()) / 2  # Each edge is stored twice
            average_length = np.mean(list(network['edge_lengths'].values()))
            loop_count = network['num_loops']
            branch_points = network['num_branches']

            picture_title = f"Image: {filename}"
            # visualize_network(network, picture_title)

            angle_tolerance = 20  # Allowed deviation for parallel judgment
            score_result = calculate_network_score(network,total_length,average_length, angle_tolerance)
            alignment_score = score_result['alignment_score']
            score = total_length - 20*loop_count - branch_points + 2*alignment_score
            score_output_path = os.path.join(output_folder, f"{base_name}_score_{score:.2f}.jpg")
            visualize_network_alignment(skeleton,network,score_result,angle_tolerance,picture_title,score,score_output_path)

            print(f"image: {filename}, score: {score:.2f}, length:{total_length:.2f}, loop:{loop_count}, branch:{branch_points}, alignment_score:{alignment_score:.2f}")

            if score > best_score:
                best_score = score
                best_image = filename
            # **Visualize extracted polylines**
            img_with_polylines = visualize_polylines(roi_gray, polylines)

            # **Save visualization of extracted polylines**
            polylines_output_path = os.path.join(output_folder, f"{base_name}_final_polylines.jpg")
            cv2.imwrite(polylines_output_path, img_with_polylines)

            # **Save simplified skeleton image**
            skeleton_output_path = os.path.join(output_folder, f"{base_name}_simplified_skeleton.jpg")
            cv2.imwrite(skeleton_output_path, skeleton)


    print(f"Best Vessel Image: {best_image}")
    print("Processing complete. Check the 'output' folder for results.")
