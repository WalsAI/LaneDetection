from ProcessingPolygons import ProcessingPolygons

def main():
    """
    Main function for preprocessing the polygons to prepare them for conversion to actual masks.
    """
    path_to_labels = '/teamspace/studios/this_studio/LaneDetection/dataset/train/labels'
    path_to_images = '/teamspace/studios/this_studio/LaneDetection/dataset/train/images'
    path_to_masks = '/teamspace/studios/this_studio/LaneDetection/dataset/train/masks'
    preprocessing_polygons = ProcessingPolygons(path_to_labels, path_to_images, path_to_masks)

    preprocessing_polygons()

if __name__ == '__main__':
    main()