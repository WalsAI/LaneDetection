class LabelToImageConverter:
    """
    Converts label title to corresponding image title.
    """
    @staticmethod
    def LabelToImage(label_title: str, image_extension: str = '.jpg') -> str:
        """
        Converts label title to corresponding image title and adding the required extension.
        If relative or absolute path is provided, only the label title will be take into consideration.

        Args:
            label_title: Title of the label.
            image_extension: Extension of the image (i.e. .jpg, .png, etc)
        """
        
        # check whether a path is provided
        if '/' in label_title:
            label_title = label_title.split('/')[-1]
            
        return '.'.join(label_title.split('.')[:-1]) + image_extension