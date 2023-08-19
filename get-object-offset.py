import cv2
import numpy as np

def calculate_distances(image, object_bounding_box):
    # Convert the object_bounding_box to a list
    object_bounding_box = list(object_bounding_box)
    
    # Get the top-left, top-right, bottom-right, and bottom-left coordinates of the object
    top_left = object_bounding_box[0]
    top_right = object_bounding_box[1]
    bottom_right = object_bounding_box[2]
    bottom_left = object_bounding_box[3]

    # Calculate the left distance
    left_distance = bottom_left[0] - top_left[0]

    # Calculate the right distance
    right_distance = bottom_right[0] - top_right[0]

    # Calculate the top distance
    top_distance = top_left[1] - bottom_left[1]

    # Calculate the bottom distance
    bottom_distance = bottom_right[1] - top_right[1]

    return left_distance, right_distance, top_distance, bottom_distance

if __name__ == "__main__":
    image = cv2.imread("3.bmp")
    object_bounding_box = [316, 316, 354, 354]
    left_distance, right_distance, top_distance, bottom_distance = calculate_distances(image, object_bounding_box)
    print("Left distance:", left_distance)
    print("Right distance:", right_distance)
    print("Top distance:", top_distance)
    print("Bottom distance:", bottom_distance)