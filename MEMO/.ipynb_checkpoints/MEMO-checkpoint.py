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

    def set_checkpoint(self, new_checkpoint_path):
        self.__checkpoint_path = new_checkpoint_path

    def save_result(self, accuracy, seed_data, path_result, num_augmentations, augmentations, seed_augmentations, top_augmentations, MEMO, fine_tuned, lr_setting):
        
        data = {
            "accuracy": accuracy,
            "seed_data": seed_data,
            "top_augmentations" : top_augmentations,
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

    def get_model(self, fine_tuned):
        model = self.__model(weights="DEFAULT") # Imagenet1k weights
        if fine_tuned:
            checkpoint = torch.load(self.__checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        model.eval()
        model.to(self.__device)
        return model 

    def get_optimizer(self, model, fine_tuned, lr_setting):
        if fine_tuned: 
            optimizer = self.__optimizer(model.parameters(), lr = 0.001)
            checkpoint = torch.load(self.__checkpoint_path)
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            layers_groups = []
            lr_base_optimizer = []
            for layers, lr_param_name in lr_setting[0].items():
                layers_groups.extend(lr_param_name[0])
                params = [param for name, param in model.named_parameters() if name in lr_param_name[0]]
                lr_base_optimizer.append({"params":params, "lr": lr_param_name[1]})
            optimizer = self.__optimizer(lr_base_optimizer, lr = lr_setting[1], momentum = 0.9, weight_decay = 0.005)
        return optimizer
        
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

        name_result = f"seed_data:{seed_data}_seed_aug:{seed_augmentations}_aug:{num_augmentations}_topaug:{top_augmentations}_MEMO:{MEMO}_finetuned:{fine_tuned}"
        path_result = os.path.join(self.__MEMO_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}" 
        if not fine_tuned: assert lr_setting, "Fine tuned is True, but no lr_setting were provided for the non-fine tuned model's optimizer"
        
        transform_loader = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        if fine_tuned:
            _, _, test_loader = get_data(batch_size, img_root, transform = transform_loader, seed = seed_data, split_data=True)
        else:
            test_loader = get_data(batch_size, img_root, transform = transform_loader, seed = seed_data, split_data=False)
            
        samples = 0.0
        cumulative_accuracy = 0.0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            for input, target in zip(inputs, targets):
                model = self.get_model(fine_tuned)
                if MEMO:
                    MEMO_augmentations = self.get_MEMO_augmentations(input, augmentations, num_augmentations, seed_augmentations)
                    optimizer = self.get_optimizer(model = model, fine_tuned = fine_tuned, lr_setting = lr_setting)
                    probab_augmentations = torch.softmax(model(MEMO_augmentations), dim = 1)
                    if top_augmentations:
                        probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                    marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                    marginal_loss = self.compute_entropy(marginal_output_distribution)
                    marginal_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                normalize_input  = T.Compose([
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input = normalize_input(input)
                probab_pred = torch.softmax(model(input.unsqueeze(0)), dim=1)
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
                         top_augmentations = top_augmentations, 
                         MEMO = MEMO, 
                         fine_tuned = fine_tuned,
                         lr_setting = lr_setting)
        
        return accuracy