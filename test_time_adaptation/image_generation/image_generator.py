import json
import os
from tqdm import tqdm
import random

import gc
import torch
import math
import ollama # if ollama is not available, install by executing the intall_and_run_ollama.sh script
from PIL import Image
import clip
import torchvision.transforms as T
import torch.nn.functional as F

class ImageGenerator:
    """
    _summary_
    """
    def __init__(self):
        """
        _summary_
        """
        pass 
        
    def get_model(self):
        print(self.__model)

    def get_text_embedding(self,clip_model, text):
        text_token = clip.tokenize(text).cuda()
        text_embedding = clip_model.encode_text(text_token).float()
        text_embedding /= text_embedding.norm()
        return text_embedding 
        
    def generate_prompts(self, num_prompts_per_class, style_of_picture, path, context_llm, llm_model = "llama3.1", clip_text_encoder = "ViT-L/14"):

        assert isinstance(llm_model,str), "Model must be a str"
        try:
          ollama.chat(llm_model)
        except ollama.ResponseError as e:
          print('Error:', e.error)
          if e.status_code == 404:
            print("Pulling the model...")
            ollama.pull(llm_model)
              
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
                    sub_dir_class = os.path.join(path, class_name)
                    prompts_to_generate = num_prompts_per_class - len(os.listdir(sub_dir_class) - 1) # -1 to account for scraped_img folder
                    if prompts_to_generate <= 0: 
                        pbar.update(1)
                        continue
                    counter_flag = 6
                    gen_prompts = []
                    original_prompts_to_gen = prompts_to_generate
                    while counter_flag>0:
                        prompts_generation_instruction = {
                            "role": "user",
                            "content": f"class:{class_name}, number of prompts:{prompts_to_generate}, style of picture: {style_of_picture}"
                        }
                        if len(context_llm) == 3:
                            context_llm.append(prompts_generation_instruction)
                        else:
                            context_llm[3] = prompts_generation_instruction
                        try:
                            response = ollama.chat(model=llm_model, messages=context_llm)
                            content = json.loads(response['message']['content'])  # json.loads to convert str to list
                            if len(content) > prompts_to_generate:
                                prompts_to_generate -= len(content)
                                gen_prompts.extend(content)
                                gen_prompts = gen_prompts[:original_prompts_to_gen]
                                counter_flag = -1
                            else:
                                prompts_to_generate -= len(content)
                                gen_prompts.extend(content)
                        except Exception as e:
                            counter_flag -= 1
    
                    if len(gen_prompts) != 1:
                        counter_flag = -1 
    
                    if counter_flag == -1:
                        num_prompts_already_gen = len(os.listdir(sub_dir_class))
                        for i in range(num_prompts_already_gen, num_prompts_already_gen + len(gen_prompts)):
                            new_sub_dir = os.path.join(path, class_name, str(i))
                            os.makedirs(new_sub_dir, exist_ok=True)
                            prompt = gen_prompts[i - num_prompts_already_gen]
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
        
    def generate_images(self, 
                        path, 
                        num_images, 
                        image_generation_pipeline, 
                        num_inference_steps, 
                        guidance_scale = 9, 
                        strength=1, 
                        clip_image_encoder = "ViT-L/14"):
        
        assert image_generation_pipeline.__class__.__name__ in ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"), "image_generation_pipeline must be one of StableDiffusionPipeline or StableDiffusionImg2ImgPipeline"

        random.seed(42)

        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(clip_image_encoder)
        clip_model.cuda().eval()

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        for class_name in tqdm(os.listdir(path), desc="Processing classes"):
            mem_allocated_before = torch.cuda.memory_allocated() 
            mem_reserved_before = torch.cuda.memory_reserved() 
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
            for gen_images_class in os.listdir(class_path):
                if num_gen_images == num_images: break
                gen_image_class = os.path.join(class_path,gen_images_class)
                with open(os.path.join(gen_image_class, "prompt.txt"), 'r') as file:
                    text_prompt = file.read()
                if image_generation_pipeline.__class__.__name__ == "StableDiffusionImg2ImgPipeline":
                    i2i_image_path = os.path.join(gen_image_class,"i2i_gen_images")
                    os.makedirs(i2i_image_path,exist_ok=True)
                    for _ in range(n_perm):
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
                            if num_gen_images == num_images: break
                else:
                    t2i_image_path = os.path.join(gen_image_class,"t2i_gen_images")
                    os.makedirs(t2i_image_path,exist_ok=True)
                    for _ in range(n_perm):
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
                            if num_gen_images == num_images: break

def retrieve_gen_images(img,  
                        num_images, 
                        clip_model, 
                        preprocess,
                        img_to_tensor_pipe,
                        data_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/imagenetA_generated",
                        use_t2i_similarity = False, 
                        t2i_images = True, 
                        i2i_images = False, 
                        threshold = 0.):

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
            image_embedding = clip_model.encode_image(preprocess(img).unsqueeze(0).cuda())
            image_embedding /= image_embedding.norm()
        
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            for gen_images_class in os.listdir(class_path):
                gen_images_class_path = os.path.join(class_path,gen_images_class)
                gen_prompt_embedding = torch.load(os.path.join(gen_images_class_path, "prompt_clip_embedding.pt"))
                t2i_similarity = F.cosine_similarity(image_embedding, gen_prompt_embedding)
                if t2i_images:
                    t2i_gen_images_main_path = os.path.join(gen_images_class_path,"t2i_gen_images")
                    try: # needed bc some prompts don't have a corresponding image yet
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
                    except:
                        pass
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

        retrieved_images = []
        for image_path in retrieved_images_paths:
            retrieved_images.append(img_to_tensor_pipe(Image.open(image_path)))
        retrieved_images = torch.stack(retrieved_images) if len(retrieved_images) >= 1 else None
            
        return retrieved_images