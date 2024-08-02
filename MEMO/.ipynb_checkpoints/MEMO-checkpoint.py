import random 
import torch
import torchvision.transforms as T
from utility.get_data import get_data, S3ImageFolder
import torchvision.models as models

import os
import json

class MEMO:
    def __init__(self, model, fine_tuned_model,optimizer, fine_tuned_optimizer, checkpoint_path, device):
        self.__model = model
        self.__fine_tuned_model = fine_tuned_model
        self.__optimizer = optimizer
        self.__fine_tuned_optimizer = fine_tuned_optimizer
        self.__checkpoint_path = checkpoint_path
        self.__device = device
        self.__MEMO_path = os.path.join(os.path.dirname(checkpoint_path), "MEMO")
        os.makedirs(self.__MEMO_path, exist_ok=True)

    def set_checkpoint(self, new_checkpoint_path):
        self.__checkpoint_path = new_checkpoint_path

    def save_result(self, accuracy, seed_data, path_result, num_augmentations, augmentations, seed_augmentations, top_k, MEMO, fine_tuned, lr_setting):
        
        data = {
            "accuracy": accuracy,
            "seed_data": seed_data,
            "top_k" : top_k,
            "use_MEMO" : MEMO,
            "fine_tuned" : fine_tuned,
            "lr_setting" : lr_setting,
            "num_augmentations" : num_augmentations,
            "seed_augmentations": seed_augmentations,
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

    def get_best_augmentations(self, probabilities, top_k):
        entropies = torch.tensor([self.compute_entropy(prob) for prob in probabilities])
        _, top_k_indices = torch.topk(entropies, top_k, largest=False, sorted=False)
        sorted_top_k_indices = top_k_indices[torch.argsort(entropies[top_k_indices])]
        top_k_probabilities = probabilities[sorted_top_k_indices]
        return top_k_probabilities
    
    def test_MEMO(self, 
                  augmentations, 
                  num_augmentations, 
                  seed_augmentations, 
                  seed_data, 
                  batch_size, 
                  img_root,
                  fine_tuned = True,
                  lr_setting = None,
                  MEMO = True,
                  top_augmentations = 0):

        name_result = f"seed_data:{seed_data}_seed_aug:{seed_augmentations}_aug:{num_augmentations}_topk:{top_k}_MEMO:{MEMO}_finetuned:{fine_tuned}"
        path_result = os.path.join(self.__MEMO_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}" 
        if not fine_tuned: assert lr_setting, "Fine tuned is True, but no lr_setting were provided for the non-fine tuned model's optimizer"
        
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
        _, _, test_loader = get_data(batch_size, img_root, transform = transform, seed = seed_data)
            
        samples = 0.0
        cumulative_accuracy = 0.0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            for input, target in zip(inputs, targets):
                if fine_tuned:
                    checkpoint = torch.load(self.__checkpoint_path)
                    self.__fine_tuned_model.load_state_dict(checkpoint["model"])
                    self.__fine_tuned_model.eval()
                    self.__fine_tuned_model.to(self.__device)
                    self.__fine_tuned_optimizer.load_state_dict(checkpoint["optimizer"])
                else:
                    base_model = self.__model(weights="DEFAULT") # Imagenet1k weights
                    base_model.eval()
                    base_model.to(self.__device)

                if MEMO:        
                    random.seed(seed_augmentations)
                    sampled_augmentations = random.sample(augmentations, num_augmentations)
                    MEMO_augmentations = torch.zeros((num_augmentations, 3, 224, 224), device=self.__device)
                    for i, augmentation in enumerate(sampled_augmentations):
                        transform_2 = T.Compose([
                                T.RandomCrop((224,224)),
                                augmentation,
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                        ])
                        augmented_input = transform_2(input.cpu()).to(self.__device) 
                        MEMO_augmentations[i] = augmented_input
                    
                    if fine_tuned:
                        for param in self.__fine_tuned_model.parameters():
                            param.requires_grad = False

                        for param in self.__fine_tuned_model.fc.parameters():
                            param.requires_grad = True

                        probab_augmentations = torch.softmax(self.__fine_tuned_model(MEMO_augmentations), dim = 1)
                        if top_augmentations:
                            probab_augmentations = self.get_best_augmentations(probab_augmentations, top_k)
                        marginal_output_distribution = torch.mean(probab_augmentations, dim=0)

                        marginal_loss = self.compute_entropy(marginal_output_distribution)
                        marginal_loss.backward()
                        self.__fine_tuned_optimizer.step()
                        self.__fine_tuned_optimizer.zero_grad()
                    else:
                        layers_groups = []
                        lr_base_optimizer = []
                        for layers, lr_param_name in lr_setting.items():
                            if layers != "other_params": 
                                layers_groups.extend(lr_param_name[0])
                                params = [param for name, param in base_model.named_parameters() if name in lr_param_name[0]]
                                lr_base_optimizer.append({"params":params, "lr": lr_param_name[1]})
                            else:
                                params = [param for name, param in base_model.named_parameters() if name not in layers_groups]
                                lr_base_optimizer.append({"params":params, "lr": lr_param_name})
                        
                        base_optimizer = self.__optimizer(lr_base_optimizer, lr = 0.001)
                        probab_augmentations = torch.softmax(base_model(MEMO_augmentations), dim = 1)
                        if top_augmentations:
                            probab_augmentations = self.get_best_augmentations(probab_augmentations, top_k)
                        marginal_output_distribution = torch.mean(probab_augmentations, dim=0)

                        marginal_loss = self.compute_entropy(marginal_output_distribution)
                        marginal_loss.backward()
                        base_optimizer.step()
                        base_optimizer.zero_grad()

                input_transform  = T.Compose([
                    T.Resize((224, 224)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input = input_transform(input)
                if fine_tuned:
                    probab_pred = torch.softmax(self.__fine_tuned_model(input.unsqueeze(0)), dim=1)
                    y_pred = probab_pred.argmax()
                    cumulative_accuracy += int(target == y_pred)
                else:
                    probab_pred = torch.softmax(base_model(input.unsqueeze(0)), dim=1)
                    y_pred = probab_pred.argmax()
                    cumulative_accuracy += int(target == y_pred)
    
            samples += inputs.shape[0]

        accuracy = cumulative_accuracy / samples * 100
        print("accuracy:", accuracy)
        self.save_result(accuracy = accuracy, 
                         seed_data = seed_data, 
                         path_result = path_result, 
                         seed_augmentations = seed_augmentations, 
                         num_augmentations = num_augmentations, 
                         augmentations = augmentations, 
                         top_k = top_k, 
                         MEMO = MEMO, 
                         fine_tuned = fine_tuned,
                         lr_setting = lr_setting)
        
        return accuracy