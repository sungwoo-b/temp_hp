import os
import sys
import json
import cv2
import numpy as np
import pandas as pd

def clip_image_around_bbox_buffer(image, bbox, buffer=100):
    """
    Clips an image around a bounding box with a buffer.

    Args:
        image (ndarray): Input image.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        buffer (int): Buffer size.

    Returns:
        ndarray or None: Clipped image.
    """
    x1, y1, x2, y2 = bbox
    x1 -= buffer
    y1 -= buffer
    x2 += buffer
    y2 += buffer
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(int(x2), image.shape[1])
    y2 = min(int(y2), image.shape[0])

    # Check for invalid clipping region
    # if x2 <= x1 or y2 <= y1:
    #     print(f"Invalid clipping region: {bbox}")
    #     return None

    clipped_image = image[y1:y2, x1:x2]
    
    return clipped_image

def main(images_dir, annotation_json, side, out_dir):  

    with open(annotation_json) as f:
        annotation_json = json.load(f)  

    os.makedirs(out_dir, exist_ok=True)

    cntr = 0

    cumulative_annos = []
    
    for image_info in annotation_json['images']:

        file_name = image_info['file_name']
        image_id = image_info['id']
        image_path = os.path.join(images_dir, file_name)
        image = cv2.imread(image_path)

        image_annotations = [anno for anno in annotation_json['annotations'] if anno['image_id'] == image_id]

        box_num = 0
        for anno in image_annotations:
            bbox = anno['bbox']
            bbxyxy = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

            # Create clipped image
            clipped_img = clip_image_around_bbox_buffer(image, bbxyxy)


            # Make output directory
            image_clipped_output_dirs = file_name.split('/')
            save_dir = os.path.join(out_dir, *image_clipped_output_dirs[:-1])
            os.makedirs(save_dir, exist_ok=True)

            # Create file name
            box_num += 1
            clipped_file_name = f"{file_name[:-4]}_{box_num}.jpg" if os.path.exists(os.path.join(out_dir, file_name)) else file_name
            save_path = os.path.join(out_dir, clipped_file_name)

            # This is a temporary fix for undetected attributes
            attributes = anno.get('attributes', {})
            image_boxes_categories = {
                "img_id": anno['image_id'], 
                "file_name": file_name, 
                "box": list(bbxyxy), 
                "image_name_clip": clipped_file_name, 
                "ann_id": anno['id'], 
                "quality": attributes.get('building_quality', "unknown"),
                "type": attributes.get('building_type', "unknown"),
                "soft": attributes.get('has_soft_story', "unknown"),
                "story": attributes.get('num_of_stories', "unknown"),
                "overhang": attributes.get('overhang_type', "unknown")
            }

            cumulative_annos.append(image_boxes_categories)
            cv2.imwrite(save_path, clipped_img)
            cntr += 1
            print(f"Annotation count: {cntr}")

    # Save CSV
    df_out = pd.DataFrame(cumulative_annos)
    csv_path = os.path.join(out_dir, f"cumulative_annos_{side}.csv")
    df_out.to_csv(csv_path, index=False)

    print(f"Saved annotation CSV: {csv_path}") # Added


# Original main function

# def main(images_dir, annotation_json, side, out_dir):  

#     annotation_json = open(annotation_json)
#     annotation_json = json.load(annotation_json)  
#     os.makedirs(out_dir, exist_ok=True)
    
#     cntr = 0

#     cumulative_annos = []
    
#     for t in annotation_json['images']:
#         tf = t['file_name']
#         ti = t['id']
#         im = cv2.imread(f"{images_dir}/{tf}")
#         box_num = 0
#         for gt in annotation_json['annotations']:
#             if gt['image_id'] == ti:
#                 bb = gt['bbox']
#                 bbxyxy = (bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3])
#                 bbox_data = list(bbxyxy)
#                 try:
#                     clipped_img = clip_image_around_bbox_buffer(im, bbox_data)
#                     if len(clipped_img) > 0:

#                         image_clipped_output_dirs = tf.split('/')
#                         os.makedirs(os.path.join(out_dir, image_clipped_output_dirs[0]), exist_ok=True)
#                         os.makedirs(os.path.join(out_dir, image_clipped_output_dirs[0], image_clipped_output_dirs[1]), exist_ok=True)
    
#                         if os.path.exists(os.path.join(out_dir, tf)):
#                             # Multiple boxes present in image
#                             box_num=box_num+1
#                             image_boxes_categories = {"img_id": gt['image_id'], "file_name": tf, "box": bbox_data, "image_name_clip": f"{tf[:-4]}_{box_num}.jpg", 
#                                                       "ann_id": gt['id'], "complete": gt['attributes']['building_completeness'],
#                                                       "condition": gt['attributes']['building_condition'],
#                                                       "material": gt['attributes']['building_material'],
#                                                       "security": gt['attributes']['building_security'],
#                                                       "use": gt['attributes']['building_use']}
#                             #print(image_boxes_categories)
#                             cumulative_annos.append(image_boxes_categories)
#                             cv2.imwrite(os.path.join(out_dir, f"{tf[:-4]}_{box_num}.jpg"), clipped_img)
#                         else:
#                             image_boxes_categories = {"img_id": gt['image_id'], "file_name": tf, "box": bbox_data, "image_name_clip": f"{tf}", 
#                                                       "ann_id": gt['id'], "complete": gt['attributes']['building_completeness'],
#                                                       "condition": gt['attributes']['building_condition'],
#                                                       "material": gt['attributes']['building_material'],
#                                                       "security": gt['attributes']['building_security'],
#                                                       "use": gt['attributes']['building_use']}
#                             #print(image_boxes_categories)
#                             cumulative_annos.append(image_boxes_categories)
#                             cv2.imwrite(os.path.join(out_dir, f"{tf}"), clipped_img)
#                         cntr = cntr+1
#                         print("Annotation count: ", cntr)
#                 except Exception as e:
#                     # Print the error message if an exception occurs
#                     print("An error occurred:", e, im, bbox_data)
#     df_out = pd.DataFrame(cumulative_annos)
#     df_out.to_csv(f"{out_dir}/cumulative_annos_{side}.csv")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python prep_classifier_training_data.py <IMG_DIR> <ANN_JSON> <SIDE> <OUT_DIR>")
        sys.exit(1)

    IMG_DIR = sys.argv[1]
    ANN_JSON = sys.argv[2] # repeat for left and right side JSON annotations
    SIDE = sys.argv[3] # repeat for left and right side JSON annotations
    OUT_DIR = sys.argv[4]

    main(IMG_DIR, ANN_JSON, SIDE, OUT_DIR)    