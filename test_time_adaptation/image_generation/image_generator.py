import torch
import json
import os
import ollama # if ollama is not available, install by executing the intall_and_run_ollama.sh script

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

    def save_image_and_embedding(self, image_tensor, embedding_tensor, file_path):
        """
        Save an image tensor and its corresponding embedding tensor in the same file.

        Args:
            image_tensor (torch.Tensor): The tensor representing the image.
            embedding_tensor (torch.Tensor): The tensor representing the embedding.
            file_path (str): The path where the file will be saved.
        """
        # Create a dictionary to store both tensors
        data = {
            'image': image_tensor,
            'embedding': embedding_tensor
        }

        # Save the dictionary to the specified file path
        torch.save(data, file_path)

    def generate_prompts(self, num_prompts, style_of_picture, path = None, context_llm = None):

        if context_llm is None:
            context_llm_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/test_time_adaptation/image_generation/llm_context.json"
            with open(file_path, 'r') as file:
                context_llm = json.load(file) 

        for class_id, class_name in classes.items():
            prompts_generation_instruction = {
                "role" : "user",
                "content" : f"class:{class_name}, number of prompts:{num_prompts}, style of picture: {style_of_picture}"
            }
            context_llm.append(prompts_generation_instruction)
            response = ollama.chat(model='llama3.1', messages=messages)
            content = response['message']['content']
            if path:
                pass
            else:
                pass
        
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