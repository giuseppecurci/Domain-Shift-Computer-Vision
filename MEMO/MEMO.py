import random 
import torch
import torchvision.transforms as T
from utility.get_data import get_data, S3ImageFolder
import torchvision.models as models

import os
import json

class MEMO:
    def __init__(self, model, optimizer, checkpoint_path, device):
        self.__model = model
        self.__optimizer = optimizer
        self.__checkpoint_path = checkpoint_path
        self.__device = device
        self.__MEMO_path = os.path.join(os.path.dirname(checkpoint_path), "MEMO")
        os.makedirs(self.__MEMO_path, exist_ok=True)
            
    def get_model(self):
        return self.__model

    def get_optimizer(self):
        return self.__optimizer

    def set_checkpoint(self, new_checkpoint_path):
        self.__checkpoint_path = new_checkpoint_path

    def save_result(self, accuracy, seed, path_result, num_augmentations, augmentations, top_k, MEMO, fine_tuned):
        data = {
            "accuracy": accuracy,
            "seed": seed,
            "top_k" : top_k,
            "use_MEMO" : MEMO,
            "fine_tuned" : fine_tuned,
            "num_augmentations" : num_augmentations,
            "augmentations" : [str(augmentation) for augmentation in augmentations]
        }
        try:
            with open(path_result, 'w') as json_file:
                json.dump(data, json_file)
        except:
            print("Result were not saved")
    
    def compute_entropy(self, probabilities):
        # Ensure probabilities are normalized (sum to 1)
        if not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
            raise ValueError("The probabilities should sum to 1.")
        
        # Compute entropy
        # Adding a small value to avoid log(0) issues
        epsilon = 1e-10
        probabilities = torch.clamp(probabilities, min=epsilon)
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        
        return entropy
    
    def test_MEMO(self, 
                  augmentations, 
                  num_augmentations, 
                  seed_augmentations, 
                  seed_data, 
                  batch_size, 
                  img_root, 
                  MEMO = True, 
                  top_k = None):

        name_result = f"seed:{seed}_aug:{num_augmentations}_topk:{top_k}_MEMO:{MEMO}_finetuned:True"
        path_result = os.path.join(self.__MEMO_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}" 
        
        torch.manual_seed(seed)

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
        # loop over zipped data loaders with different transform
        _, _, test_loader = get_data(batch_size, img_root, transform = transform, seed = seed)
    
        checkpoint = torch.load(self.__checkpoint_path)
        
        samples = 0.0
        cumulative_accuracy = 0.0
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            for input, target in zip(inputs, targets):
                self.__model.load_state_dict(checkpoint["model"])
                self.__model.eval()
                self.__model.to(self.__device) 

                if MEMO:
                    for param in self.__model.parameters():
                        param.requires_grad = False
    
                    for param in self.__model.fc.parameters():
                        param.requires_grad = True

                    self.__optimizer.load_state_dict(checkpoint["optimizer"])
                    
                    random.manual_seed(seed)
                    sampled_augmentations = random.sample(augmentations, num_augmentations)
                    MEMO_augmentations = torch.zeros((num_augmentations, 3, 224, 224), device=self.__device) # *input.shape
                    for i, augmentation in enumerate(sampled_augmentations):
                        transform_2 = T.Compose([
                                T.RandomCrop((224,224)),
                                augmentation,
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                        ])
                        augmented_input = transform_2(input.cpu()).to(self.__device) 
                        MEMO_augmentations[i] = augmented_input
                    
                    probab_augmentations = torch.softmax(self.__model(MEMO_augmentations), dim = 1)
                    marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                    
                    marginal_loss = self.compute_entropy(marginal_output_distribution)
                    marginal_loss.backward()
                    self.__optimizer.step()
                    self.__optimizer.zero_grad()

                input_transform  = T.Compose([
                    T.Resize((224, 224)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input = input_transform(input)
                probab_pred = torch.softmax(self.__model(input.unsqueeze(0)), dim=1)
                y_pred = probab_pred.argmax()
                cumulative_accuracy += int(target == y_pred)
    
            samples += inputs.shape[0]

        accuracy = cumulative_accuracy / samples * 100
        print("accuracy:", accuracy)
        self.save_result(accuracy, seed, path_result, num_augmentations, augmentations, top_k, MEMO, fine_tuned = True)
        
        return accuracy

    def test_MEMO_imagenet1k(self, 
                             augmentations, 
                             num_augmentations, 
                             seed_augmentations, 
                             seed_data, 
                             batch_size, 
                             img_root, 
                             MEMO = True, 
                             top_k = None):

        name_result = f"seed:{seed}_aug:{num_augmentations}_topk:{top_k}_MEMO:{MEMO}_finetuned:False"
        path_result = os.path.join(self.__MEMO_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}" 
        
        torch.manual_seed(seed)

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
        # loop over zipped data loaders with different transform
        _, _, test_loader = get_data(batch_size, img_root, transform = transform, seed = seed)
            
        samples = 0.0
        cumulative_accuracy = 0.0
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            for input, target in zip(inputs, targets):
                model = models.resnet50(weights="IMAGENET1K_V1")
                model.eval()
                model.to(self.__device)
                if MEMO:
                    for param in model.parameters():
                        param.requires_grad = False
    
                    for param in model.fc.parameters():
                        param.requires_grad = True
                        
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.005)

                    random.manual_seed(seed)
                    sampled_augmentations = random.sample(augmentations, num_augmentations)
                    MEMO_augmentations = torch.zeros((num_augmentations, 3, 224, 224), device=self.__device) # *input.shape
                    for i, augmentation in enumerate(sampled_augmentations):
                        transform_2 = T.Compose([
                                T.RandomCrop((224,224)),
                                augmentation,
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                        ])
                        augmented_input = transform_2(input.cpu()).to(self.__device) 
                        MEMO_augmentations[i] = augmented_input
                    
                    probab_augmentations = torch.softmax(model(MEMO_augmentations), dim = 1)
                    marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                    
                    marginal_loss = self.compute_entropy(marginal_output_distribution)
                    marginal_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                input_transform  = T.Compose([
                    T.Resize((224, 224)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input = input_transform(input)
                probab_pred = torch.softmax(model(input.unsqueeze(0)), dim=1)
                y_pred = probab_pred.argmax()
                cumulative_accuracy += int(target == y_pred)
    
            samples += inputs.shape[0]

        accuracy = cumulative_accuracy / samples * 100
        print("accuracy:", accuracy)
        self.save_result(accuracy, seed, path_result, num_augmentations, augmentations, top_k, MEMO, fine_tuned = False)
        
        return accuracy   