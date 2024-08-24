import torch
import json
import os

from ollama import generate # if ollama is not available, install by executing the intall_and_run_ollama.sh script

class Tester:
    """
    _summary_
    """
    def __init__(self, LLM):
        """
        _summary_
        """
        self._LLM = LLM

    def get_imagenetA_classes(self):
        """
        ImageNet-A uses the same label structure as the original ImageNet (ImageNet-1K).
        Each class in ImageNet is represented by a synset ID (e.g., n01440764 for "tench, Tinca tinca").
        This function returns a dictionary that maps the synset IDs of ImageNet-A to the corresponding class names.
        ----------
        indices_in_1k: list of indices to map [B,1000] -> [B,200]
        """
        imagenetA_classes_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/utility/data/imagenetA_classes.json"
        imagenetA_classes_dict = None
        with open(imagenetA_classes_path, 'r') as json_file:
            imagenetA_classes_dict = json.load(json_file)

        # ensure `class_dict` is a dictionary with keys as class IDs and values as class names
        class_dict = {k: v for k, v in imagenetA_classes_dict.items()}
        return class_dict

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

    # to be completed
    def generate_images(self):
        """
        _summary_

        Args:
            LM (_type_): _description_
        """
        data_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/utility/data/st_images"
        os.makedirs(data_path, exist_ok=True)

        classes = self.get_imagenetA_classes()

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