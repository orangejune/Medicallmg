from Analysis_ReadImg import *

from Analysis_Extraction import *
from Analysis_MaskColor import *

from Analysis_Distance import *

full_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/Series-001/jpg-img-00001-00003.jpg"  # Replace with your grayscale image path
blue_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/Series-001/jpg-img-00001-00004.jpg"  # Replace with your image with blue lines

# full_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-4LAD原始.dcm"  # Replace with your grayscale image path
# blue_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-5LAD描记.dcm"  # Replace with your image with blue lines

##todo: use a pair of images -- every pair onrigial and lableded
full_image_path = r"KD/New/冠脉瘤1-wangziyi/KD-CAL-wangziyi/1-左冠脉整体.dcm"  # Replace with your grayscale image path
blue_image_path = r"KD/New/冠脉瘤1-wangziyi/KD-CAL-wangziyi/2-LCA LAD描记.dcm"  # Replace with your image with blue lines

##todo: mask parameter
hsv_ranges = {
    "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    # "green": [(np.array([40, 100, 100]), np.array([70, 255, 255]))],
    # "blue": [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))] ##todo: better not use yellow
}
target_bbox = (160, 70, 910, 500)  # Bounding box (left, top, right, bottom)
buffer_size = 15  # Example buffer size in pixels
maskcnt_min = 10
##todo: extraction parameter
# Parameters for boundary extraction
intensity_range = (90, 180)  # Target intensity range for the organ
morph_kernel = (10, 10)         # Morphological kernel size
min_area = 500                # Minimum area to keep a contour
bbox = (160, 70, 910, 500)    # Bounding box (left, top, right, bottom)


##todo: read img
readimg = Analysis_ReadImg(full_image_path, blue_image_path, True)
##todo: extract mask
# Initialize the extractor
extractor = MaskBufferExtractor(readimg.full_image, readimg.blue_image)
# Input parameters
# todo:Process each color to produce a buffer zone
for color, ranges in hsv_ranges.items():
    print(f"Processing color: {color}")
    # Create mask restricted to the bounding box
    mask, maskcnt = extractor.create_color_mask(extractor.blue_image, mask_name=color, hsv_ranges=ranges, bbox=target_bbox)
    if maskcnt < maskcnt_min: ##todo: skip if the color mask is not large
        continue
    # todo:Create a buffer around the mask
    buffered_mask = extractor.create_buffered_mask(mask_name=color, buffer_size=buffer_size)

    # Display results
    cv2.imshow(f"{color.capitalize()} Mask", mask)
    cv2.imshow(f"{color.capitalize()} Buffered Mask", buffered_mask )
    cv2.waitKey(0)

    ce = ContourExtraction(readimg.full_image, readimg.blue_image)
    for i_pixel_min in [90,]: ##todo: not using the threshold for extraction seems OTUS works the best
        for i_pixel_max in [200]:
            for i_kernel in [3,]:
                for i_area in [100,]: ##todo: helpful to elimate small regions like image rulers
                    valid_parts_contours, full_contours, segment_contours =  ce.extract_boundaries_and_filter_by_mask(buffered_mask, morph_kernel=(i_kernel, i_kernel),
                                                                                        min_area=i_area, min_threshold=i_pixel_min, max_threshold=i_pixel_max)
                    ce.visualize_contours(full_contours,valid_parts_contours, segment_contours)

                    restricted_contours = valid_parts_contours
                    ##todo: create a distance computation object
                    distance_check = DistanceChecker(restricted_contours, readimg.full_image)
                    distances, nearest_points, max_diameter, max_points, sampled_contour1, sampled_contour2 = distance_check.debug_distances(
                        num_points=1000)

                    # Step 4: Visualize and save results
                    distance_check.visualize_debug_distances(sampled_contour1, sampled_contour2, distances,
                                                             nearest_points, max_diameter, max_points,
                                                             save_path=f"KD/{color}_distance_{i_pixel_min}_{i_pixel_max}_{i_kernel}_{i_area}_visualization.png")

##todo: check random connection-- topology; check regional range; can support multiple colors but different output images --need to combine





