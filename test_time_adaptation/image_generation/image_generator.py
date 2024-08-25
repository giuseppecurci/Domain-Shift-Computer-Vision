import json
import os
from tqdm import tqdm
import random

import torch
import math
import ollama # if ollama is not available, install by executing the intall_and_run_ollama.sh script
from PIL import Image
import clip

class ImageGenerator:
    """
    _summary_
    """
    def __init__(self, model):
        """
        _summary_
        """
        assert isinstance(model,str), "Model must be a str"
        try:
          ollama.chat(model)
        except ollama.ResponseError as e:
          print('Error:', e.error)
          if e.status_code == 404:
            print("Pulling the model...")
            ollama.pull(model)
        self.__model = model

    def get_model(self):
        print(self.__model)

    def get_text_embedding(self,clip_model, text):
        text_token = clip.tokenize(text).cuda()
        text_embedding = clip_model.encode_text(text_token).float()
        text_embedding /= text_embedding.norm()
        return text_embedding 
        
    def generate_prompts(self, num_prompts, style_of_picture, path, context_llm, clip_text_encoder = "ViT-L/14"):
        
        if isinstance(context_llm,str):
            with open(context_llm, 'r') as file:
                context_llm = json.load(file) 

        skipped_classes = []

        clip_model, _ = clip.load(clip_text_encoder)
        clip_model.cuda().eval()
        
        for class_name in tqdm(os.listdir(path), desc="Processing classes"):
            prompts_generation_instruction = {
                "role": "user",
                "content": f"class:{class_name}, number of prompts:{num_prompts}, style of picture: {style_of_picture}"
            }
        
            if len(context_llm) == 3:
                context_llm.append(prompts_generation_instruction)
            else:
                context_llm[3] = prompts_generation_instruction
            
            counter_flag = 3
            while counter_flag>0:
                try:
                    response = ollama.chat(model=self.__model, messages=context_llm)
                    content = json.loads(response['message']['content'])  # json.loads to convert str to list
                    assert len(content) == num_prompts, (
                        "The model failed to generate the number of prompts requested. "
                        "Try to run the command again or consider changing context_llm"
                    )
                    counter_flag = -1
                except Exception as e:
                    counter_flag -= 1

            if counter_flag == -1:
                sub_dir_class = os.listdir(os.path.join(path, class_name))
                for i in range(len(sub_dir_class), num_prompts + len(sub_dir_class)):
                    new_sub_dir = os.path.join(path, class_name, str(i))
                    os.makedirs(new_sub_dir, exist_ok=True)
                    prompt = content[i - len(sub_dir_class)]
                    prompt_embedding = self.get_text_embedding(clip_model, prompt)
                    with open(os.path.join(new_sub_dir, "prompt.txt"), 'w') as file:
                        file.write(prompt)
                    torch.save(prompt_embedding, os.path.join(new_sub_dir,"prompt_clip_embedding.pt"))
            else:
                skipped_classes.append(class_name)
                print(f"Skipping class {class_name}.")

        return skipped_classes

    def get_image_embedding(self,clip_model, preprocess, image):
        image_preprocessed = preprocess(image).unsqueeze(0).cuda()
        image_embedding = clip_model.encode_image(image_preprocessed)
        image_embedding /= image_embedding.norm()
        return image_embedding
        
    def generate_images(self, path, num_images, image_generation_pipeline, num_inference_steps, guidance_scale = 9, strength=1, clip_image_encoder = "ViT-L/14"):
        
        assert image_generation_pipeline.__class__.__name__ in ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"), "image_generation_pipeline must be one of StableDiffusionPipeline or StableDiffusionImg2ImgPipeline"

        random.seed(42)

        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(clip_image_encoder)
        clip_model.cuda().eval()
        
        for class_name in tqdm(os.listdir(path), desc="Processing classes"):
            class_path = os.path.join(path, class_name)
            num_prompts = len(os.listdir(class_path))
            if image_generation_pipeline.__class__.__name__ == "StableDiffusionImg2ImgPipeline":
                print("Loading scraped images...")
                scraped_image_paths = os.path.join(class_path, "scraped_images")
                scraped_images = []
                for scraped_image_path in scraped_image_paths:
                    scraped_image = Image.open(scraped_image_path)
                    scraped_image = scraped_image.resize((512,512))
                    scraped_images.append(scraped_image)
            num_prompts = len(os.listdir(class_path))
            n_perm = math.ceil(num_images / num_prompts)
            num_gen_images = 0 # needed if num_images < num_prompts
            for gen_images_class in tqdm(os.listdir(class_path), desc="Generating images"):
                gen_image_class = os.path.join(class_path,gen_images_class)
                with open(os.path.join(gen_image_class, "prompt.txt"), 'r') as file:
                    text_prompt = file.read()
                if image_generation_pipeline.__class__.__name__ == "StableDiffusionImg2ImgPipeline":
                    i2i_image_path = os.path.join(gen_image_class,"i2i_gen_images")
                    os.makedirs(i2i_image_path,exist_ok=True)
                    for _ in tqdm(range(n_perm), desc="Generating Img2Img images"):
                        scraped_image = random.sample(scraped_images,1)[0]
                        with torch.no_grad():
                            gen_image = image_generation_pipeline(prompt=text_prompt,
                                                                  image=scraped_image,
                                                                  strength=strength,
                                                                  guidance_scale=guidance_scale,
                                                                  num_inference_steps=num_inference_steps).images[0]
                        gen_image_embedding = self.get_image_embedding(clip_model, preprocess, gen_image)
                        save_gen_image_path = os.path.join(i2i_image_path,str(len(os.listdir(i2i_image_path))))
                        os.makedirs(save_gen_image_path)
                        torch.save(gen_image_embedding, os.path.join(save_gen_image_path, "image_embedding.pt"))
                        gen_image.save(os.path.join(save_gen_image_path, "image.png"))
                        del gen_image, gen_image_embedding
                        torch.cuda.empty_cache()
                        num_gen_images += 1
                        if num_gen_images > num_images: break
                else:
                    t2i_image_path = os.path.join(gen_image_class,"t2i_gen_images")
                    os.makedirs(t2i_image_path,exist_ok=True)
                    for _ in tqdm(range(n_perm), desc="Generating Text2Img images"):
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
                        del gen_image, gen_image_embedding
                        torch.cuda.empty_cache()
                        num_gen_images += 1
                        if num_gen_images > num_images: break
                if num_gen_images > num_images: break