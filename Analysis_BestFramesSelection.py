import os
from Analysis_FrameCycleDetection import *
from Analysis_FrameSplit import *
from Analysis_FrameSelection import *

def find_top3_in_peaks(scores, peak_frames, window_size=3):
    """
    Within the index range of all peak_frames and their surrounding window_size,
    find the top 3 global maximum values and their indices.
    
    Parameters:
        scores: List or array of scores
        peak_frames: List of peak indices
        window_size: Size of the window (how many indices to take before and after each peak)
        
    Returns:
        top3_values: Top 3 largest values in descending order
        top3_indices: Corresponding indices of these values
    """
    all_indices = set()
    
    for peak in peak_frames:
        start = max(0, peak - window_size)
        end = min(len(scores), peak + window_size + 1)
        all_indices.update(range(start, end))
    
    all_indices = list(all_indices)
    candidate_scores = [scores[i] for i in all_indices]
    
    scored_indices = list(zip(candidate_scores, all_indices))
    
    scored_indices.sort(reverse=True, key=lambda x: x[0])
    
    top3 = scored_indices[:3]
    
    top3_values = [x[0] for x in top3]
    top3_indices = [x[1] for x in top3]
    
    return top3_values, top3_indices

if __name__=="__main__":

    # Folder paths (modify as needed)
    media_names = ['Media1', 'Media2', 'Media3', 'Media6', 'Media13', 'Media17', 'Media18', 'Media19', 'Media21']
    for media_name in media_names:
        video_path = f"./heart_cycles/{media_name}.mp4"
        input_folder = f"./heart_cycles/{media_name}"
        output_folder = f"./heart_cycles/{media_name}_Score"
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.isdir(input_folder) or not os.listdir(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            extract_frames(video_path, input_folder, frame_interval=1)

        cycle_frames, peak_frames = detect_heartbeat_cycles(video_path)    

        # Define ROI coordinates (top-left and bottom-right)
        roi_top_left = (110, 100)
        roi_bottom_right = (640, 500)

        close_gap_threshold = 10  # Maximum distance to close gaps between polylines
        best_score = -np.inf
        best_image = None
        scores = []
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
                
                scores.append(score)
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

        top3_values, top3_indices = find_top3_in_peaks(scores, peak_frames, window_size=3)

        print("\nDetected Peak Frames (Contractions):", peak_frames)
        print(f"{media_name} Top 3 values: {top3_values}")
        print(f"{media_name} Top 3 indices: {top3_indices}")
        print()
    print("Processing complete. Check the 'output' folder for results.")