import cv2

def draw_bboxes(camera_bboxes, result_image):
    # Print bounding box on top of the panoptic labels
    for bbox in camera_bboxes:
        label = bbox.type
        bbox_coords = bbox.box
        br_x = int(bbox_coords.center_x + bbox_coords.length/2)
        br_y = int(bbox_coords.center_y + bbox_coords.width/2)
        tl_x = int(bbox_coords.center_x - bbox_coords.length/2)
        tl_y = int(bbox_coords.center_y - bbox_coords.width/2)

        tl = (tl_x, tl_y)
        br = (br_x, br_y)

        result_image = cv2.rectangle(result_image, tl, br, (255,0,0), 2)