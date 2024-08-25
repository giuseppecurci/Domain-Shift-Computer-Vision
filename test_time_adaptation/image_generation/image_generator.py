import torch
import json
import os
import ollama # if ollama is not available, install by executing the intall_and_run_ollama.sh script
from tqdm import tqdm
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
        text_token = clip.tokenize("I like fish").cuda()
        text_embedding = clip_model.encode_text(text_token).float()
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
            
    def generate_images(self, num_images):
        """
        _summary_

        Args:
            LM (_type_): _description_
        """
        assert isinstance(num_images,int) and num_images >= 1, "num_samples must be >1 and int" 
        data_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/utility/data/st_images"
        os.makedirs(data_path, exist_ok=True)
        
        # Generate prompt sequences for each class in ImageNet
        prompts = {}
        for class_id, class_name in classes.items():
            # Generate a response using the LLM model
            question = f"Can you generate {self._LLM['num_samples']} sentences having as subject {class_name}?"
            response = generate(self._LLM['model_name'], question)
            response = response['response']

            # NOT TESTED
            # Extract the prompt sequences from the response
            sequences = [item['text'] for item in response]

            # Create a directory for the current class
            class_dir = os.path.join(data_path, class_id)
            os.makedirs(class_dir, exist_ok=True)

            # Generate images for each prompt sequence
            for i, sequence in enumerate(sequences):
                # Generate an image for the current prompt sequence
                image = None  # Replace this with the code to generate an image from the prompt sequence
                embed = None # Replace this with the code to generate an embedding from the prompt sequence

                # Save the image and embedding to a file
                file_path = os.path.join(class_dir, f'image_{i}.pt')
                self._save_image_and_embedding(image, embed, file_path)