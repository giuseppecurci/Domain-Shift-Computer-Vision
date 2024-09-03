import json
import os
from tqdm import tqdm
import random
from typing import Union, Dict, List
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import torch
import torchvision
import math
import ollama # if ollama is not available, install by executing the intall_and_run_ollama.sh script
from PIL import Image
import clip
import torchvision.transforms as T
import torch.nn.functional as F

class ImageGenerator:
    """
    A class containing all the functions to generate and save prompts, images and their respective
    CLIP embeddings.
    """
    def __init__(self):
        pass 

    def get_text_embedding(self,clip_model, text: str):
        """
        
        """
        text_token = clip.tokenize(text).cuda()
        text_embedding = clip_model.encode_text(text_token).float()
        text_embedding /= text_embedding.norm()
        return text_embedding 
        
    def generate_prompts(self, 
                         num_prompts_per_class: int, 
                         style_of_picture: str, 
                         path: str, 
                         context_llm: Union[str, Dict], 
                         llm_model: str = "llama3.1", 
                         clip_text_encoder: str = "ViT-L/14",
                         img_classes: List[str] = []
                        ) -> None:
        """
        Generate image prompts for each class in the given directory path using a language model available in ollama library.
        The prompts will then be used to generate images using a diffusion model.
        ---
        num_prompts_per_class (int): Number of prompts to generate per class. This number doesn't correspond to the actual number 
                                     of prompts that will be generated, but rather to the desired total number of prompts for each 
                                     class e.g. if the class "ant" has already 10 prompts and num_prompts_per_class = 12, then only
                                     two prompts will be generated.
        style_of_picture (str): Style to be used in image prompts.
        path (str): Path to the generated dataset.
        context_llm (str, dict): A dict containing the context for the language model or a path to a json file containing it.
        llm_model (str): The language model to use for generating prompts.
        clip_text_encoder (str): The CLIP text encoder model to use. 
        ---
        Returns:
            list: A list of class names for which prompts could not be generated due to bad prompts formatting.
        """
        assert isinstance(llm_model,str), "Model must be a str"
        assert isinstance(context_llm, (str,dict)), "context_llm must be a str"
        assert isinstance(clip_text_encoder, str), "clip_text_encoder must be a str. Use clip.available_models() to get valid strings."
        assert isinstance(style_of_picture, str), "style_of_picture must be a str representing the style of the image that will be generated"
        assert isinstance(num_prompts_per_class, int), "num_prompts_per_class must be an int"

        # check that llm_model is available
        try:
          ollama.chat(llm_model)
        except ollama.ResponseError as e:
          print('Error:', e.error)
          if e.status_code == 404:
            # try to pull the model if it exists
            print("Pulling the model...")
            ollama.pull(llm_model)

        # load context_llm dict if needed
        if isinstance(context_llm,str):
            with open(context_llm, 'r') as file:
                context_llm = json.load(file) 

        skipped_classes = []

        clip_model, _ = clip.load(clip_text_encoder)
        clip_model.cuda().eval()

        class_list = os.listdir(path)
        with torch.no_grad():
            with tqdm(total=len(class_list), desc="Processing classes") as pbar:
                for class_name in class_list:
                    pbar.set_description(f"Processing class: {class_name}")
                    if img_classes and class_name not in img_classes: 
                        pbar.update(1)
                        continue 
                    sub_dir_class = os.path.join(path, class_name)
                    prompts_to_generate = num_prompts_per_class - len(os.listdir(sub_dir_class) - 1) # -1 to account for scraped_img folder
                    if prompts_to_generate <= 0: 
                        pbar.update(1)
                        continue
                    # sometimes the language model doesn't return an appropriate output, tolerance = number of possible attempts
                    tolerance = 6
                    gen_prompts = []
                    original_prompts_to_gen = prompts_to_generate
                    while tolerance>0 and prompts_to_generate>0:
                        prompts_generation_instruction = {
                            "role": "user",
                            "content": f"class:{class_name}, number of prompts:{prompts_to_generate}, style of picture: {style_of_picture}"
                        }
                        if len(context_llm) == 3:
                            context_llm.append(prompts_generation_instruction)
                        else:
                            # needed from the second iteration
                            context_llm[3] = prompts_generation_instruction
                        try:
                            response = ollama.chat(model=llm_model, messages=context_llm)
                            content = json.loads(response['message']['content'])  # json.loads to convert str to list
                            prompts_to_generate -= len(content)
                            gen_prompts.extend(content)
                            gen_prompts = gen_prompts[:original_prompts_to_gen]
                        except Exception as e:
                            tolerance -= 1
    
                    if gen_prompts:
                        num_prompts_already_gen = len(os.listdir(sub_dir_class))
                        for i in range(num_prompts_already_gen, num_prompts_already_gen + len(gen_prompts)):
                            new_sub_dir = os.path.join(path, class_name, str(i))
                            os.makedirs(new_sub_dir, exist_ok=True)
                            prompt = gen_prompts[i - num_prompts_already_gen]
                            prompt_embedding = self.get_text_embedding(clip_model, prompt) # compute text CLIP embedding
                            with open(os.path.join(new_sub_dir, "prompt.txt"), 'w') as file:
                                file.write(prompt)
                            torch.save(prompt_embedding, os.path.join(new_sub_dir,"prompt_clip_embedding.pt"))
                    
                    if len(gen_prompts) < original_prompts_to_gen:
                        # at least some prompts were not generated
                        skipped_classes.append(class_name)
                        print(f"Skipping class {class_name}.")

        return skipped_classes
    
    def get_image_embedding(self,clip_model, preprocess, image):
        image_preprocessed = preprocess(image).unsqueeze(0).cuda()
        image_embedding = clip_model.encode_image(image_preprocessed)
        image_embedding /= image_embedding.norm()
        return image_embedding
        
    def generate_images(self, 
                        path: str, 
                        num_images_per_class: int, 
                        class_to_skip: List[str], 
                        image_generation_pipeline: Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline], 
                        num_inference_steps: int, 
                        guidance_scale: int = 9, 
                        strength: float = 0.8, 
                        clip_image_encoder: str = "ViT-L/14"
                       ) -> None:
        """
        Generate images for each class in the specified directory using the given image generation pipeline.
        ---
        path (str): Path to the directory containing class subdirectories.
        num_images_per_class (int): Number of images to generate per class. 
        class_to_skip (List[str]): List of classes to skip.
        image_generation_pipeline (Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]): The image generation pipeline to use.
        num_inference_steps (int): Number of inference steps for image generation.
        guidance_scale (int): Guidance scale for image generation. A higher guidance scale value encourages the model to generate 
                                        images closely linked to the text prompt at the expense of lower image quality
        strength (float): Indicates extent to transform the reference scraped image. A value of 1 essentially ignores image. 
        clip_image_encoder (str, optional): The CLIP image encoder model to use.
        ---
        Returns:
            None: This function does not return any value.
        """
        assert isinstance(image_generation_pipeline, (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)), \
            "image_generation_pipeline must be one of StableDiffusionPipeline or StableDiffusionImg2ImgPipeline"
        assert isinstance(clip_image_encoder, str), "clip_image_encoder must be a str. Use clip.available_models() to get valid strings."
        assert isinstance(num_images_per_class, int), "num_images_per_class must be an int"
        
        random.seed(42)

        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(clip_image_encoder)
        clip_model.cuda().eval()

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        class_list = os.listdir(path)
        with torch.no_grad():
            with tqdm(total=len(class_list), desc="Processing classes") as pbar:
                for class_name in class_list:
                    pbar.set_description(f"Processing class: {class_name}")
                    if class_name in class_to_skip: 
                        pbar.update(1)
                        continue
                    class_path = os.path.join(path, class_name)
                    num_prompts = len(os.listdir(class_path))
                    
                    if image_generation_pipeline.__class__.__name__ == "StableDiffusionImg2ImgPipeline":
                        print("Loading scraped images...")
                        scraped_image_paths = os.path.join(class_path, "scraped_images") # go in scraped_images folder
                        scraped_images = []
                        for scraped_image_path in os.listdir(scraped_image_paths): # open and append scraped images
                            if scraped_image_path in (".ipynb_checkpoints"): continue
                            img_path = os.path.join(scraped_image_paths,scraped_image_path)
                            scraped_image = Image.open(img_path)
                            scraped_image = scraped_image.resize((512,512))
                            scraped_images.append(scraped_image)
                    
                    num_gen_images = 0 # needed if num_images < num_prompts
    
                    while num_gen_images < num_images_per_class:
                        for gen_images_class in os.listdir(class_path):
                            if gen_images_class in (".ipynb_checkpoints","scraped_images"): continue
                            gen_image_class = os.path.join(class_path,gen_images_class)
                            # needed bc some folders don't have a prompt.txt due to some error during generation
                            try:
                                with open(os.path.join(gen_image_class, "prompt.txt"), 'r') as file:
                                    text_prompt = file.read()
                            except:
                                continue
                                
                            # get scraped images using the image-to-image generation pipeline
                            if isinstance(image_generation_pipeline, StableDiffusionImg2ImgPipeline):
                                try: # sometimes get weird OOM error
                                    i2i_image_path = os.path.join(gen_image_class,"i2i_gen_images")
                                    os.makedirs(i2i_image_path,exist_ok=True)
                                    scraped_image = random.sample(scraped_images,1)[0]
                                    with torch.no_grad():
                                        gen_image = image_generation_pipeline(prompt=text_prompt,
                                                                                image=scraped_image,
                                                                                strength=strength,
                                                                                guidance_scale=guidance_scale,
                                                                                num_inference_steps=num_inference_steps).images[0]
                                        del scraped_image
                                        gen_image_embedding = self.get_image_embedding(clip_model, preprocess, gen_image)
                                        save_gen_image_path = os.path.join(i2i_image_path,str(len(os.listdir(i2i_image_path))))
                                        os.makedirs(save_gen_image_path)
                                        torch.save(gen_image_embedding, os.path.join(save_gen_image_path, "image_embedding.pt"))
                                        gen_image.save(os.path.join(save_gen_image_path, "image.png"))
                                        num_gen_images += 1
                                except:
                                    print("Error occurred")
                            else:
                                t2i_image_path = os.path.join(gen_image_class,"t2i_gen_images")
                                os.makedirs(t2i_image_path,exist_ok=True)
                                with torch.no_grad():
                                    gen_image = image_generation_pipeline(prompt=text_prompt,
                                                                            strength=strength,
                                                                            guidance_scale=guidance_scale,
                                                                            num_inference_steps=num_inference_steps).images[0]
                                    gen_image_embedding = self.get_image_embedding(clip_model, preprocess, gen_image)
                                    save_gen_image_path = os.path.join(t2i_image_path,str(len(os.listdir(t2i_image_path))))
                                    os.makedirs(save_gen_image_path)
                                    torch.save(gen_image_embedding, os.path.join(save_gen_image_path, "image_embedding.pt"))
                                    gen_image.save(os.path.join(save_gen_image_path, "image.png"))                        
                                    num_gen_images += 1
                            # break loop over the class prompts if generated enough images
                            if num_gen_images == num_images_per_class: break 
                    pbar.update(1)

def retrieve_gen_images(img: Union[torch.Tensor, Image.Image],  
                        num_images: int,
                        clip_model: clip.model.CLIP, 
                        clip_preprocess: torchvision.transforms.transforms.Compose,
                        img_to_tensor_pipe: torchvision.transforms.transforms.Compose = None,
                        data_path: str = "Domain-Shift-Computer-Vision/imagenetA_generated",
                        use_t2i_similarity: bool = False, 
                        t2i_images: bool = True,
                        i2i_images: bool = False, 
                        threshold: float = 0.0) -> Union[torch.Tensor, None]:
        """
        Retrieve the most similar generated images based on CLIP embeddings.
        ---
        img (Union[torch.Tensor, Image.Image]): The input image to compare against. Can be a torch tensor or PIL image.
        num_images (int): The number of similar images to retrieve.
        clip_model: The preloaded CLIP model for generating embeddings.
        clip_preprocess: The preprocessing function for the CLIP model.
        img_to_tensor_pipe: A pipeline function that converts images to tensors.
        data_path (str): Path to the directory containing generated images.
        use_t2i_similarity (bool): Whether to average text-to-image similarity with image-to-image similarity.
        t2i_images (bool): Whether to include text-to-image generated images in the search.
        i2i_images (bool): Whether to include image-to-image generated images in the search.
        threshold (float): The minimum cosine similarity threshold for an image to be considered. 
        ---
        Returns:
            Union[torch.Tensor, None]: A tensor containing the retrieved images. Returns None if no images are retrieved.
        """
        assert i2i_images or t2i_images, "One of t2i_images and i2i_images must be true"
        assert isinstance(use_t2i_similarity, bool), "use_t2i_similarity must be a bool"
        assert isinstance(t2i_images, bool), "t2i_images must be a bool"
        assert isinstance(i2i_images, bool), "i2i_images must be a bool"
        assert isinstance(num_images, int), "num_images must be an int"
        assert isinstance(threshold, float) and 0 < threshold < 1, "threshold must be a float and between 0 and 1"
        
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        retrieved_images_paths = []
        retrieved_images_similarity = torch.zeros(num_images)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(clip_preprocess(img).unsqueeze(0).cuda())
            image_embedding /= image_embedding.norm()
        
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            for gen_images_class in os.listdir(class_path):
                if gen_images_class in ["scraped_images", ".ipynb_checkpoints"]: continue
                gen_images_class_path = os.path.join(class_path,gen_images_class)
                gen_prompt_embedding = torch.load(os.path.join(gen_images_class_path, "prompt_clip_embedding.pt"))
                t2i_similarity = F.cosine_similarity(image_embedding, gen_prompt_embedding)
                # Search in text-to-image generated images
                if t2i_images:
                    t2i_gen_images_main_path = os.path.join(gen_images_class_path,"t2i_gen_images")
                    for t2i_images_paths in os.listdir(t2i_gen_images_main_path):
                        t2i_image_path = os.path.join(t2i_gen_images_main_path,t2i_images_paths)
                        gen_image_embedding = torch.load(os.path.join(t2i_image_path, "image_embedding.pt"))
                        i2i_similarity = F.cosine_similarity(image_embedding, gen_image_embedding)
                        if use_t2i_similarity:
                            similarity = (i2i_similarity + t2i_similarity)/2 # avg similarity
                        else:
                            similarity = i2i_similarity
                        if similarity < threshold: continue
                        if len(retrieved_images_paths) < num_images:
                            retrieved_images_similarity[len(retrieved_images_paths)] = similarity
                            retrieved_images_paths.append(os.path.join(t2i_image_path, "image.png"))
                        else:
                            min_similarity, id_similarity = retrieved_images_similarity.min(dim=0)
                            if similarity > min_similarity:
                                retrieved_images_similarity[id_similarity] = similarity
                                retrieved_images_paths[id_similarity] = os.path.join(t2i_image_path, "image.png")
                # Search in image-to-image generated images
                if i2i_images:
                    i2i_gen_images_main_path = os.path.join(gen_images_class_path,"i2i_gen_images")
                    for i2i_images_paths in os.listdir(i2i_gen_images_main_path):
                        i2i_image_path = os.path.join(i2i_gen_images_main_path,i2i_images_paths)
                        gen_image_embedding = torch.load(os.path.join(i2i_image_path, "image_embedding.pt"))
        
                        i2i_similarity = F.cosine_similarity(image_embedding, gen_image_embedding)
                        if use_t2i_similarity:
                            similarity = (i2i_similarity + t2i_similarity)/2 # avg similarity
                        else:
                            similarity = i2i_similarity
                        if similarity < threshold: continue
                        if len(retrieved_images_paths) < num_images:
                            retrieved_images_similarity[len(retrieved_images_paths)] = similarity
                            retrieved_images_paths.append(os.path.join(t2i_image_path, "image.png"))
                        else:
                            min_similarity, id_similarity = retrieved_images_similarity.min(dim=0)
                            if similarity > min_similarity:
                                retrieved_images_similarity[id_similarity] = similarity
                                retrieved_images_paths[id_similarity] = os.path.join(i2i_image_path, "image.png")
        
        # Load and return the retrieved images as a tensor
        retrieved_images = []
        for image_path in retrieved_images_paths:
            if img_to_tensor_pipe:
                retrieved_images.append(img_to_tensor_pipe(Image.open(image_path)))
            else:
                retrieved_images.append(Image.open(image_path))

        if img_to_tensor_pipe:
            retrieved_images = torch.stack(retrieved_images) if len(retrieved_images) >= 1 else torch.tensor([])
        return retrieved_images