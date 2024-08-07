import random 
import torch
import torchvision.transforms as T
from utility.get_data import get_data, S3ImageFolder
import torchvision.models as models

import os
import json

class MEMO:
    def __init__(self, model, optimizer, exp_path, device):
        self.__model = model
        self.__optimizer = optimizer
        self.__device = device
        self.__MEMO_path = os.path.join(exp_path, "MEMO")
        os.makedirs(self.__MEMO_path, exist_ok=True)

    def set_checkpoint(self, new_checkpoint_path):
        self.__checkpoint_path = new_checkpoint_path

    def save_result(self, accuracy, seed_data, path_result, num_augmentations, augmentations, seed_augmentations, top_augmentations, MEMO, fine_tuned, lr_setting, weights_imagenet):
        
        data = {
            "accuracy": accuracy,
            "top_augmentations" : top_augmentations,
            "use_MEMO" : MEMO,
            "lr_setting" : lr_setting,
            "weights_imagenet" : str(weights_imagenet),
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

    def get_MEMO_augmentations(self, input, augmentations, num_augmentations, seed_augmentations):
        random.seed(seed_augmentations)
        sampled_augmentations = random.sample(augmentations, num_augmentations)
        MEMO_augmentations = torch.zeros((num_augmentations, 3, 224, 224), device=self.__device)
        for i, augmentation in enumerate(sampled_augmentations):
            transform_MEMO = T.Compose([
                T.ToPILImage(),
                augmentation,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            augmented_input = transform_MEMO(input.cpu()).to(self.__device)
            MEMO_augmentations[i] = augmented_input
        return MEMO_augmentations

    def get_model(self, weights_imagenet):
        model = self.__model(weights=weights_imagenet)
        model.eval()
        model.to(self.__device)
        return model 

    def get_optimizer(self, model, lr_setting):
        if len(lr_setting) == 2:
            layers_groups = []
            lr_optimizer = []
            for layers, lr_param_name in lr_setting[0].items():
                layers_groups.extend(lr_param_name[0])
                params = [param for name, param in model.named_parameters() if name in lr_param_name[0]]
                lr_optimizer.append({"params":params, "lr": lr_param_name[1]})
            other_params = [param for name, param in model.named_parameters() if name not in layers_groups]
            lr_optimizer.append({"params":other_params})
            optimizer = self.__optimizer(lr_optimizer, lr = lr_setting[1], momentum = 0.9, weight_decay = 0)
        else:
            optimizer = self.__optimizer(models.resnet50().parameters(), lr = lr_setting, momentum = 0.9, weight_decay = 0)
        return optimizer

    def get_imagenetA_masking(self):
        imagenetA_masking_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/MEMO/imagenetA_masking.json"
        with open(imagenetA_masking_path, 'r') as json_file:
            imagenetA_masking = json.load(json_file)
        indices_in_1k = [int(k) for k in imagenetA_masking if imagenetA_masking[k] != -1]
        return indices_in_1k

    def test_MEMO(self, 
                  augmentations, 
                  num_augmentations, 
                  seed_augmentations,  
                  img_root,
                  weights_imagenet,
                  lr_setting,
                  dataset = "imagenetA",
                  batch_size = 64,
                  MEMO = True,
                  top_augmentations = 0,
                  verbose = True,
                  log_interval = 1):

        weights_name = str(weights_imagenet).split(".")[-1]
        name_result = f"model:{self.__model.__name__}_weights:{weights_name}_seed_aug:{seed_augmentations}_aug:{num_augmentations}_topaug:{top_augmentations}_MEMO:{MEMO}"
        path_result = os.path.join(self.__MEMO_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}" 
        
        transform_loader = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # to use after
        normalize_input  = T.Compose([
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        
        test_loader = get_data(batch_size, img_root, transform = transform_loader, split_data=False)
            
        model = self.get_model(weights_imagenet)
        if dataset == "imagenetA":
            imagenetA_masking = self.get_imagenetA_masking()
            
        samples = 0.0
        cumulative_accuracy = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            if MEMO:
                for input, target in zip(inputs, targets):
                    model = self.get_model(weights_imagenet)
                    MEMO_augmentations = self.get_MEMO_augmentations(input, augmentations, num_augmentations, seed_augmentations)
                    optimizer = self.get_optimizer(model = model, lr_setting = lr_setting)
                    logits = model(MEMO_augmentations)
                    logits = logits[:, imagenetA_masking]
                    probab_augmentations = torch.softmax(logits, dim=1)
                    if top_augmentations:
                        probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                    marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                    marginal_loss = self.compute_entropy(marginal_output_distribution)
                    marginal_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    input = normalize_input(input)
                    probab_pred = torch.softmax(model(input.unsqueeze(0)), dim=1)
                    y_pred = probab_pred.argmax()
                    cumulative_accuracy += int(target == y_pred)
            else:
                with torch.no_grad():
                    inputs = normalize_input(inputs)
                    logits = model(inputs)
                    logits = logits[:,imagenetA_masking]
                    predicted = torch.argmax(logits, dim=1)
                    cumulative_accuracy += (predicted == targets).sum().item()
                    
            samples += inputs.shape[0]

            if verbose and batch_idx % log_interval == 0:
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(test_loader)}, Accuracy: {current_accuracy:.2f}%', end='\r')

        accuracy = cumulative_accuracy / samples * 100
        self.save_result(accuracy = accuracy, 
                         path_result = path_result, 
                         seed_augmentations = seed_augmentations, 
                         num_augmentations = num_augmentations, 
                         augmentations = augmentations, 
                         top_augmentations = top_augmentations, 
                         MEMO = MEMO, 
                         lr_setting = lr_setting,
                         weights_imagenet = weights_imagenet)
        
        return accuracy