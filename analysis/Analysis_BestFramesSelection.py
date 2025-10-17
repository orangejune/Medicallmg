import os
from Analysis_FrameCycleDetection import *
from utils.Analysis_FrameSplit import *
from Analysis_FrameSelection import *
from ultralytics import YOLO
import torch
"""
打分：yolo训练后的置信度/骨架评分
"""
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

def get_score(video_path, input_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)
    score_img_path = f'{output_folder}/yolo_imgs'
    os.makedirs(score_img_path, exist_ok=True)

    if not os.path.isdir(input_folder) or not os.listdir(input_folder):
        os.makedirs(input_folder, exist_ok=True)
        extract_frames(video_path, input_folder, frame_interval=1)

    cycle_frames, peak_frames = detect_heartbeat_cycles(video_path)    

    close_gap_threshold = 10  # Maximum distance to close gaps between polylines
    best_score = -np.inf
    best_image = None
    scores = []
    pred_image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    for filename in pred_image_files:
        img_path = os.path.join(input_folder, filename)
        results = model.predict(source=img_path, save=False, imgsz=640, device=device)
        img = cv2.imread(img_path)

        base_name = os.path.splitext(filename)[0]  # Extract filename without extension

        roi = []
        conf = 0
        cls = None
        r = results[0]
        for box in r.boxes:
            if conf < box.conf.item():
                roi = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = box.conf.item()
        print(base_name, cls, conf)


        if roi == []:
            continue
        x1, y1, x2, y2 = [int(i) for i in roi]
        # Convert to grayscale and apply ROI
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y1:y2, x1:x2]

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


        score_img_output_path = os.path.join(score_img_path, f'score{score}_conf{conf:.2f}_{base_name}.jpg')
        cv2.imwrite(score_img_output_path, img)

    # top3_values, top3_indices = find_top3_in_peaks(scores, peak_frames, window_size=3)

    # print("\nDetected Peak Frames (Contractions):", peak_frames)
    # print(f"{media_name} Top 3 values: {top3_values}")
    # print(f"{media_name} Top 3 indices: {top3_indices}")
    # print()

if __name__=="__main__":

    # Folder paths (modify as needed)
    media_names = ['Image08','Image14','Image19']
    script_full_path = os.path.abspath(__file__)
    script_directory = 'yolo_score_result'
    MODEL_BEST_PATH = r'C:\Users\june.lin\Desktop\medicallmg\runs\detect\train11\weights\best.pt'
    model = YOLO(MODEL_BEST_PATH)
    if torch.cuda.is_available():
        device = 0  # 使用第一块GPU
        print(f"GPU is available. Using device: cuda:{device}")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    else:
        device = 'cpu' # 如果没有GPU，则使用CPU
        print("GPU not found. Using device: cpu. Training will be slow.")
    for media_name in media_names:
        video_path = f"{script_directory}/{media_name}.avi"
        input_folder = f"{script_directory}/{media_name}"
        output_folder = f"{script_directory}/{media_name}_Score"
        get_score(video_path, input_folder, output_folder, model, device)
    print("Processing complete. Check the 'output' folder for results.")