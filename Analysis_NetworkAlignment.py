import numpy as np
import math
from collections import defaultdict
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import random

def calculate_network_score(network, total_length, average_length, angle_tolerance=20):
    """
    Calculate the parallelism score of the network structure
    Parameters:
        network: Skeleton network data
        angle_tolerance: Allowed angle deviation (degrees)
    Returns:
        score: Network score (higher indicates better parallel structure)
        avg_angle: Average angle of the network (degrees)
        aligned_length: Total length of edges that meet the angle criteria
        total_length: Total length of the network
    """
    # Calculate the weighted average angle of all edges
    angle_sum = 0
    n = 0

    for (u, v), length in network['edge_lengths'].items():
        if u < v:  # Avoid duplicate outputs
            path = network['edge_paths'][(u, v)]
            if len(path) < 2:
                continue

            if length > average_length:
                # start = np.array(path[0])
                # end = np.array(path[-1])
                start, end = [np.array(path[0]), np.array(path[-1])] if np.array(path[0])[1] < np.array(path[-1])[1] else [np.array(path[-1]), np.array(path[0])]
                dx = end[1] - start[1]  # x corresponds to the column coordinate in the image
                dy = end[0] - start[0]  # y corresponds to the row coordinate in the image
                angle = math.degrees(math.atan2(dy, dx))

                angle_sum += angle
                n += 1
            else:
                continue

    # Calculate the weighted average angle
    avg_angle = angle_sum / n

    # Calculate the network score
    score = 0
    aligned_length = 0

    for (u, v), length in network['edge_lengths'].items():
        if u < v:  # Avoid duplicate outputs
            path = network['edge_paths'][(u, v)]
            if len(path) < 2:
                continue

            start, end = [np.array(path[0]), np.array(path[-1])] if np.array(path[0])[1] < np.array(path[-1])[1] else [
                np.array(path[-1]), np.array(path[0])]
            dx = end[1] - start[1]
            dy = end[0] - start[0]
            edge_angle = math.degrees(math.atan2(dy, dx))

            angle_diff = abs(edge_angle - avg_angle)

            if angle_diff <= angle_tolerance:
                score += length
                aligned_length += length
            else:
                score -= length

    # Normalize the score to a 0-1 range (optional)
    normalized_score = (score + total_length) / (2 * total_length) if total_length > 0 else 0

    return {
        'alignment_score': score,
        'normalized_score': normalized_score,
        'average_angle': avg_angle,
        'aligned_length': aligned_length,
        'total_length': total_length,
        'alignment_ratio': aligned_length / total_length if total_length > 0 else 0
    }


def visualize_network_alignment(image, network, network_score, angle_tolerance, picture_title, total_score, score_output_path):
    avg_angle = network_score['average_angle']
    """Visualize the alignment of network angles"""
    img_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(12, 10))

    # Draw all edges
    for (u, v), path in network['edge_paths'].items():
        if u < v:  # Avoid duplicate outputs
            path_arr = np.array(path)
            # Calculate the angle of the current edge
            start, end = [path_arr[0], path_arr[-1]] if path_arr[0, 1] < path_arr[-1, 1] else [path_arr[-1], path_arr[0]]
            dx = end[1] - start[1]
            dy = end[0] - start[0]
            edge_angle = math.degrees(math.atan2(dy, dx))

            angle_diff = abs(edge_angle - avg_angle)

            # Choose color based on alignment
            if angle_diff <= angle_tolerance:
                color = (0, 255, 0)  # Green indicates well-aligned edges
            else:
                color = (255, 0, 0)  # Red indicates misaligned edges

            # Draw the edge
            plt.plot(path_arr[:, 1], path_arr[:, 0], color=np.array(color) / 255, linewidth=2)

    # Draw the average direction indicator line
    center = np.array([image.shape[0] // 2, image.shape[1] // 2])
    arrow_length = min(image.shape) * 0.4
    end_point = center + np.array([
        arrow_length * math.sin(math.radians(avg_angle)),
        arrow_length * math.cos(math.radians(avg_angle))
    ])

    plt.arrow(center[1], center[0],
              end_point[1] - center[1], end_point[0] - center[0],
              color='yellow', linewidth=4, head_width=10)

    # Format score information
    score_info = (
        f"Total Score: {total_score:.2f}\n"
        f"Num of Loops: {network['num_loops']}\n"
        f"Num of Branches: {network['num_branches']}\n"
        f"Total Length: {network_score['total_length']:.2f} px\n"
        f"Aligned Length: {network_score['aligned_length']:.2f} px\n"
        f"Average Angle: {network_score['average_angle']:.1f}°\n"
        f"Alignment Score: {network_score['alignment_score']:.2f}\n"
        f"Alignment Ratio: {network_score['alignment_ratio']:.2f}"
    )
    # Add score information
    plt.text(0.05, 0.95, score_info, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(picture_title)
    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(score_output_path)
    plt.close()
