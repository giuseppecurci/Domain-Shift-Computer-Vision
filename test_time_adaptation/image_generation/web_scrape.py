from bing_image_downloader import downloader
import os
from tqdm import tqdm

def scrape_images_imagenetA(img_style: str, imgenetA_gen_path: str, num_images = 5):
    """
    Scrape images for each imagenet-A class in the given directory path using the specified image style.
    Images are intially stored in a folder named as "img_style + \" \" class_name". Then the folder is renamed
    scraped_images.
    ---
    img_style (str): The style or keywords to use for image queries.
    imgenetA_gen_path (str): The path to the directory containing class subdirectories.
    limit (int, optional): The minimum number of images to scrape per class. Default is 5.
    """
    class_list = os.listdir(imgenetA_gen_path) # get classes' name
    with tqdm(total=len(class_list), desc="Processing classes") as pbar:
        for class_name in class_list:
            pbar.set_description(f"Processing class: {class_name}")
            class_path = os.path.join(imgenetA_gen_path, class_name)
            new_scraped_img_path = os.path.join(class_path, "scraped_images")
            # if for a class enough images have already been retrieved then skip it
            if os.path.exists(new_scraped_img_path):
                if len(os.listdir(new_scraped_img_path)) >= num_images: 
                    pbar.update(1)
                    continue 
            query = img_style + " " + class_name
            downloader.download(query = query, 
                                limit=num_images, 
                                output_dir=class_path, 
                                adult_filter_off=True, 
                                force_replace=False, 
                                timeout=60,
                                verbose=False)
            current_scraped_img_path = os.path.join(class_path, query)
            os.rename(current_scraped_img_path, new_scraped_img_path)
            pbar.update(1)