import json 
import os 

def get_imagenetA_classes():
    """
    ImageNet-A uses the same label structure as the original ImageNet (ImageNet-1K).
    Each class in ImageNet is represented by a synset ID (e.g., n01440764 for "tench, Tinca tinca").
    This function returns a dictionary that maps the synset IDs of ImageNet-A to the corresponding class names.
    ----------
    indices_in_1k: list of indices to map [B,1000] -> [B,200]
    """
    imagenetA_classes_path = "Domain-Shift-Computer-Vision/utility/data/imagenetA_classes.json"
    imagenetA_classes_dict = None
    with open(imagenetA_classes_path, 'r') as json_file:
        imagenetA_classes_dict = json.load(json_file)

    # ensure `class_dict` is a dictionary with keys as class IDs and values as class names
    class_dict = {k: v for k, v in imagenetA_classes_dict.items()}
    return class_dict

def create_dir_generated_images(path):    
    classes = list(get_imagenetA_classes().values())
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        os.makedirs(class_path, exist_ok=True)