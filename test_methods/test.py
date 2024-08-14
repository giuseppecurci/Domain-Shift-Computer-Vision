import torch
import torchvision.transforms as T
import torch.nn.functional as F

from utility.data.get_data import get_data
from test_time_adaptation.adaptive_bn import adaptive_bn_forward
from test_time_adaptation.MEMO import compute_entropy, get_best_augmentations, get_test_augmentations

import os
import json

class Tester:
    """
    A class to run all the experiments. It stores all the informations to reproduce the experiments in a json file 
    at exp_path. 
    """
    def __init__(self, model, optimizer, exp_path, device):
        self.__model = model
        self.__optimizer = optimizer
        self.__device = device
        self.__exp_path = exp_path

    def save_result(self, accuracy, path_result, num_augmentations, augmentations, seed_augmentations, top_augmentations, MEMO, lr_setting, weights, prior_strength):
        """
        Takes all information of the experiment saves it in a json file stored at exp_path
        """
        data = {
            "accuracy": accuracy,
            "top_augmentations" : top_augmentations,
            "use_MEMO" : MEMO,
            "lr_setting" : lr_setting,
            "weights" : weights,
            "num_augmentations" : num_augmentations,
            "seed_augmentations": seed_augmentations,
            "augmentations" : [str(augmentation) for augmentation in augmentations],
            "prior_strength" : prior_strength
        }
        try:
            with open(path_result, 'w') as json_file:
                json.dump(data, json_file)
        except:
            print("Result were not saved")

    def get_model(self, weights_imagenet):
        """
        Utility function to instantiate a torch model. The argument weights_imagenet should have 
        a value in accordance with the parameter weights of torchvision.models.
        """
        model = self.__model(weights=weights_imagenet)
        model.to(self.__device)
        model.eval()
        return model 

    def get_optimizer(self, model, lr_setting:list):
        """
        Utility function to instantiate a torch optimizer.
        ----------
        lr_setting: must be a list containing either one global lr for the whole model or a dictionary 
        where each value is a list with a list of parameters' names and a lr for those parameters. 
        e.g. 
        lr_setting = [{
            "classifier" : [["fc.weight", "fc.bias"], 0.00025]    
            }, 0]
        lr_setting = [0.00025]
        """
        if len(lr_setting) == 2:
            layers_groups = []
            lr_optimizer = []
            for layers, lr_param_name in lr_setting[0].items():
                layers_groups.extend(lr_param_name[0])
                params = [param for name, param in model.named_parameters() if name in lr_param_name[0]]
                lr_optimizer.append({"params":params, "lr": lr_param_name[1]})
            other_params = [param for name, param in model.named_parameters() if name not in layers_groups]
            lr_optimizer.append({"params":other_params})
            optimizer = self.__optimizer(lr_optimizer, lr = lr_setting[1], weight_decay = 0)
        else:
            optimizer = self.__optimizer(model.parameters(), lr = lr_setting[0], weight_decay = 0)
        return optimizer

    def get_imagenetA_masking(self):
        """
        All torchvision models output a tensor [B,1000] with "B" being the batch dimension. This function 
        returns a list of indices to apply to the model's output to use the model on imagenet-A dataset.
        ----------
        indices_in_1k: list of indices to map [B,1000] -> [B,200]
        """
        imagenetA_masking_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/utility/data/imagenetA_masking.json"
        with open(imagenetA_masking_path, 'r') as json_file:
            imagenetA_masking = json.load(json_file)
        indices_in_1k = [int(k) for k in imagenetA_masking if imagenetA_masking[k] != -1]
        return indices_in_1k

    def get_prediction(self, image_tensors, model, masking, TTA = False, top_augmentations = 0):
        """
        Takes a tensor of images and outputs a prediction for each image.
        ----------
        image_tensors: is a tensor of [B,C,H,W] if TTA is used or if both MEMO and TTA are not used, or of dimension [C,H,W]
                       if only MEMO is used
        masking: a list of indices to map the imagenet1k logits to the one of imagenet-A
        top_augmentations: a non-negative integer, if greater than 0 then a the "top_augmentations" with the lowest entropy are
                           selected to make the final prediction 
        """
        logits = model(image_tensors)[:,masking] if image_tensors.dim() == 4 else model(image_tensors.unsqueeze(0))[:,masking]
        if TTA:
            probab_augmentations = F.softmax(logits - logits.max(dim=1)[0][:, None], dim=1)
            if top_augmentations:
                probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
            y_pred = probab_augmentations.mean(dim=0).argmax().item()
            return y_pred
        return logits.argmax(dim=1)

    def compute_entropy(self, probabilities: torch.tensor):
        """
        See MEMO.py
        """
        return compute_entropy(probabilities)

    def get_best_augmentations(self, probabilities: torch.tensor, top_k: int):
        """
        See MEMO.py
        """
        return get_best_augmentations(probabilities, top_k)

    def get_test_augmentations(self, input:torch.tensor, augmentations:list, num_augmentations:int, seed_augmentations:int):
        """
        See MEMO.py
        """
        return get_test_augmentations(input, augmentations, num_augmentations, seed_augmentations)
        
    def test(self, 
             augmentations:list, 
             num_augmentations:int, 
             seed_augmentations:int,  
             img_root:str,
             lr_setting:list,
             weights_imagenet = None,
             dataset = "imagenetA",
             batch_size = 64,
             MEMO = False,
             top_augmentations = 0,
             TTA = False,
             prior_strength = -1,
             verbose = True,
             log_interval = 1):
        """
        Main function to test a torchvision model with different test-time adaptation techniques 
        and keep track of the results and the experiment setting. 
        ---
        augmentations: list of torchvision.transforms functions.
        num_augmentations: the number of augmentations to use for each sample to perform test-time adaptation.
        seed_augmentations: seed to reproduce the sampling of augmentations.
        img_root: str path to get a dataset in a torch format.
        lr_setting: list with lr instructions to adapt the model. See "get_optimizer" for more details.
        weights_imagenet: weights_imagenet should have a value in accordance with the parameter 
                          weights of torchvision.models.
        dataset: the name of the dataset to use. Note: this parameter doesn't directly control the data 
                 used, it's only used to use the right masking to map the models' outputs to the right dimensions. 
                 At the moment only Imagenet-A masking is supported.
        MEMO: a boolean to use marginal entropy minimization with one test point
        TTA: a boolean to use test time augmentation
        top_augmentations: if MEMO or TTA are set to True, then values higher than zero select the top_augmentations 
                           with the lowest entropy (highest confidence).
        prior_strength: defines the weight given to pre-trained statistics in BN adaptation. If negative, then no BN
                        adaptation is applied.
        verbose: use loading bar to visualize accuracy and number of batch during testing.
        log_interval: defines after how many batches a new accuracy should be displayed. Default is 1, thus 
                      after each batch a new value is displayed. 
        """
        # check some basic conditions
        if not (MEMO or TTA):
            assert not (num_augmentations or top_augmentations), "If both MEMO and TTA are set to False, then top_augmentations must be 0"
        assert not (weights_imagenet or lr_setting) if not MEMO else True, "If MEMO is false, then lr_setting and weights_imagenet must be None" 
        assert prior_strength >= 0, "prior_strength must a non-negative float"
        assert isinstance(prior_strength, (float,int)) , "Prior adaptation must be either a float or an int"
        
        # get the name of the weigths used and define the name of the experiment 
        weights_name = str(weights_imagenet).split(".")[-1] if weights_imagenet else "MEMO_repo"
        name_result = f"MEMO:{MEMO}_adaptBN:{prior_strength}_TTA:{TTA}_aug:{num_augmentations}_topaug:{top_augmentations}_seed_aug:{seed_augmentations}_weights:{weights_name}"
        path_result = os.path.join(self.__exp_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}"

        # transformation pipeline used in ResNet-50 original training
        transform_loader = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])

        # to use after model's update
        normalize_input  = T.Compose([
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        
        test_loader = get_data(batch_size, img_root, transform = transform_loader, split_data=False)
        model = self.get_model(weights_imagenet)

        # if MEMO is used, create a checkpoint to reload after each model and optimizer update
        if MEMO:
            optimizer = self.get_optimizer(model = model, lr_setting = lr_setting)
            MEMO_checkpoint_path = os.path.join(self.__exp_path,"checkpoint.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, MEMO_checkpoint_path)
            MEMO_checkpoint = torch.load(MEMO_checkpoint_path)
        
        if dataset == "imagenetA":
            imagenetA_masking = self.get_imagenetA_masking()

        if prior_strength < 0:
            torch.nn.BatchNorm2d.prior_strength = 1
        else:
            torch.nn.BatchNorm2d.prior_strength = prior_strength / (prior_strength + 1)
            torch.nn.BatchNorm2d.forward = adaptive_bn_forward
            
        samples = 0.0
        cumulative_accuracy = 0.0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.__device), targets.to(self.__device)
            if MEMO or TTA:
                for input, target in zip(inputs, targets):
                    if MEMO:
                        model.load_state_dict(MEMO_checkpoint['model'])
                        model.eval()
                        optimizer.load_state_dict(MEMO_checkpoint['optimizer'])

                    # get normalized augmentations
                    test_augmentations = self.get_test_augmentations(input, augmentations, num_augmentations, seed_augmentations)
                    test_augmentations = test_augmentations.to(self.__device)
                    logits = model(test_augmentations)

                    # apply imagenetA masking
                    if dataset == "imagenetA":
                        logits = logits[:, imagenetA_masking] 
                    # compute stable softmax
                    probab_augmentations = F.softmax(logits - logits.max(dim=1)[0][:, None], dim=1) 

                    # confidence selection for augmentations
                    if top_augmentations:
                        probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                    
                    if MEMO:
                        marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                        marginal_loss = self.compute_entropy(marginal_output_distribution)
                        marginal_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    with torch.no_grad():
                        if TTA:
                            y_pred = self.get_prediction(test_augmentations, model, imagenetA_masking, TTA, top_augmentations)
                        else:
                            input = normalize_input(input)
                            y_pred = self.get_prediction(input, model, imagenetA_masking)
                        cumulative_accuracy += int(target == y_pred)
            else:
                with torch.no_grad():
                    inputs = normalize_input(inputs)
                    y_pred = self.get_prediction(inputs, model, imagenetA_masking)
                    cumulative_accuracy += (y_pred == targets).sum().item()
                    
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
                         weights = weights_name,
                         prior_strength = prior_strength)
        
        return accuracy