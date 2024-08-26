import torch
import torchvision.transforms as T
import torch.nn.functional as F

from utility.data.get_data import get_data
from test_time_adaptation.adaptive_bn import adaptive_bn_forward
from test_time_adaptation.MEMO import compute_entropy, get_best_augmentations, get_test_augmentations
from test_time_adaptation.resnet50_dropout import ResNet50Dropout

import os
import json
import time
from scipy import stats
from PIL import Image

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

    def save_result(self, accuracy, path_result, num_augmentations, augmentations, seed_augmentations, top_augmentations, MEMO, num_adaptation_steps, lr_setting, weights, prior_strength, time_test, use_MC):
        """
        Takes all information of the experiment saves it in a json file stored at exp_path
        """
        data = {
            "accuracy": accuracy,
            "top_augmentations" : top_augmentations,
            "use_MEMO" : MEMO,
            "num_adaptation_steps" : num_adaptation_steps,
            "lr_setting" : lr_setting,
            "weights" : weights,
            "num_augmentations" : num_augmentations,
            "seed_augmentations": seed_augmentations,
            "augmentations" : [str(augmentation) for augmentation in augmentations],
            "prior_strength" : prior_strength,
            "MC" : use_MC,
            "time_test" : time_test
        }
        try:
            with open(path_result, 'w') as json_file:
                json.dump(data, json_file)
        except:
            print("Result were not saved")

    def get_model(self, weights_imagenet, MC):
        """
        Utility function to instantiate a torch model. The argument weights_imagenet should have
        a value in accordance with the parameter weights of torchvision.models.
        """
        if MC:
            self.__model=ResNet50Dropout(weights=weights_imagenet, dropout_rate=MC['dropout_rate'])
            model = self.__model
        else:
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

    def get_monte_carlo_statistics(self, mc_logits):
        """
        Compute mean, median, mode and standard deviation of the Monte Carlo samples.
        """
        statistics = {}
        mean_logits = mc_logits.mean(dim=0)
        statistics['mean'] = mean_logits

        median_logits = mc_logits.median(dim=0).values
        statistics['median'] = median_logits

        pred_classes = mc_logits.argmax(dim=1)
        pred_classes_cpu = pred_classes.cpu().numpy()
        mode_predictions, _ = stats.mode(pred_classes_cpu, axis=0)
        mode_predictions = torch.tensor(mode_predictions.squeeze(), dtype=torch.long)
        statistics['mode'] = mode_predictions

        uncertainty = mc_logits.var(dim=0)
        statistics['std'] = uncertainty
        return statistics

    def get_prediction(self, image_tensors, model, masking, TTA = False, top_augmentations = 0, MC = None):
        """
        Takes a tensor of images and outputs a prediction for each image.
        ----------
        image_tensors: is a tensor of [B,C,H,W] if TTA is used or if both MEMO and TTA are not used, or of dimension [C,H,W]
                       if only MEMO is used
        masking: a list of indices to map the imagenet1k logits to the one of imagenet-A
        top_augmentations: a non-negative integer, if greater than 0 then the "top_augmentations" with the lowest entropy are
                           selected to make the final prediction
        MC: a dictionary containing the number of evaluations using Monte Carlo Dropout and the dropout rate
        """
        if MC:
            model.train()  # enable dropout by setting the model to training mode
            mc_logits = []
            for _ in range(MC['num_samples']):
                logits = model(image_tensors)[:,masking] if image_tensors.dim() == 4 else model(image_tensors.unsqueeze(0))[:,masking]
                mc_logits.append(logits)
            mc_logits = torch.stack(mc_logits, dim=0)
            if TTA:
                # first mean is over MC samples, second mean is over TTA augmentations
                probab_augmentations = F.softmax(mc_logits - mc_logits.max(dim=2, keepdim=True)[0], dim=2)
                if top_augmentations:
                    probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                y_pred = probab_augmentations.mean(dim=0).mean(dim=0).argmax().item()
                statistics = self.get_monte_carlo_statistics(probab_augmentations.mean(dim=1))
                return y_pred, statistics
            statistics = self.get_monte_carlo_statistics(mc_logits)
            return statistics['median'].argmax(dim=1), statistics
        else:
            logits = model(image_tensors)[:,masking] if image_tensors.dim() == 4 else model(image_tensors.unsqueeze(0))[:,masking]
            if TTA:
                probab_augmentations = F.softmax(logits - logits.max(dim=1)[0][:, None], dim=1)
                if top_augmentations:
                    probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                y_pred = probab_augmentations.mean(dim=0).argmax().item()
                return y_pred, None
            return logits.argmax(dim=1), None

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

    def retrieve_generated_images(img, num_images, clip_model, preprocess, img_to_tensor_pipe, data_path, use_t2i_similarity, t2i_images, i2i_images,
                                  threshold):
        """
        See image_generator.py.
        """
        return retrieve_gen_images(img,  
                                   num_images, 
                                   clip_model, 
                                   preprocess,
                                   img_to_tensor_pipe,
                                   data_path = "/home/sagemaker-user/Domain-Shift-Computer-Vision/imagenetA_generated",
                                   use_t2i_similarity = False, 
                                   t2i_images = True, 
                                   i2i_images = False, 
                                   threshold = 0.)
    
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
             num_adaptation_steps = 0,
             top_augmentations = 0,
             TTA = False,
             prior_strength = -1,
             verbose = True,
             log_interval = 1,
             MC = None,
             use_generated_augmentations = False):
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
        assert bool(num_adaptation_steps) == MEMO, "When using MEMO adaptation steps should be > 1, otherwise equal to 0."  
        if not (MEMO or TTA):
            assert not (num_augmentations or top_augmentations), "If both MEMO and TTA are set to False, then top_augmentations and num_augmentations must be 0"
        assert not lr_setting if not MEMO else True, "If MEMO is false, then lr_setting must be None" 
        assert isinstance(prior_strength, (float,int)) , "Prior adaptation must be either a float or an int"

        # get the name of the weigths used and define the name of the experiment 
        weights_name = str(weights_imagenet).split(".")[-1] if weights_imagenet else "MEMO_repo"
        use_MC = True if MC else False
        name_result = f"MEMO_{MEMO}_AdaptSteps_{num_adaptation_steps}_adaptBN_{prior_strength}_TTA_{TTA}_aug_{num_augmentations}_topaug_{top_augmentations}_seed_aug_{seed_augmentations}_weights_{weights_name}_MC_{use_MC}"
        path_result = os.path.join(self.__exp_path,name_result)
        assert not os.path.exists(path_result),f"MEMO test already exists: {path_result}"

        # in case of using dropout, check if the model is a ResNet50Dropout and the parameters are correct
        if MC:
            assert isinstance(self.__model, ResNet50Dropout), f"To use dropout the model must be a ResNet50Dropout"
            assert MC['num_samples'] > 1, f"To use dropout the number of samples must be greater than 1" 

        # transformation pipeline used in ResNet-50 original training
        transform_loader = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])

        # to use after model's update
        normalize_input = T.Compose([
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        test_loader = get_data(batch_size, img_root, transform = transform_loader, split_data=False)
        model = self.get_model(weights_imagenet, MC)

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

        # Initialize a dictionary to store accumulated time for each step
        time_dict = {
            "MEMO_update": 0.0,
            "get_augmentations": 0.0,
            "confidence_selection": 0.0,
            "get_prediction": 0.0,
            "total_time": 0.0
        }

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
                    start_time_augmentations = time.time()
                    test_augmentations = self.get_test_augmentations(input, augmentations, num_augmentations, seed_augmentations)
                    end_time_augmentations = time.time()
                    time_dict["get_augmentations"] += (end_time_augmentations - start_time_augmentations)

                    if use_generated_augmentations:
                        test_augmentations = test_augmentations.to(self.__device)
                    
                    for _ in range(num_adaptation_steps):
                        logits = model(test_augmentations)
    
                        # apply imagenetA masking
                        if dataset == "imagenetA":
                            logits = logits[:, imagenetA_masking]
                        # compute stable softmax
                        probab_augmentations = F.softmax(logits - logits.max(dim=1)[0][:, None], dim=1)

                        # confidence selection for augmentations
                        if top_augmentations:
                            start_time_confidence_selection = time.time()
                            probab_augmentations = self.get_best_augmentations(probab_augmentations, top_augmentations)
                            end_time_confidence_selection = time.time()
                            time_dict["confidence_selection"] += (end_time_confidence_selection - start_time_confidence_selection)

                        if MEMO:
                            start_time_memo_update = time.time()
                            marginal_output_distribution = torch.mean(probab_augmentations, dim=0)
                            marginal_loss = self.compute_entropy(marginal_output_distribution)
                            marginal_loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            end_time_memo_update = time.time()
                            time_dict["MEMO_update"] += (end_time_memo_update - start_time_memo_update)

                    start_time_prediction = time.time()
                    with torch.no_grad():
                        if TTA:
                            # statistics:
                            # dictionary containing statistics resulting from the application of monte carlo dropout
                            # look at get_monte_carlo_statistics() for more details
                            y_pred, statistics = self.get_prediction(test_augmentations, model, imagenetA_masking, TTA, top_augmentations, MC=MC)
                        else:
                            input = normalize_input(input)
                            y_pred, statistics = self.get_prediction(input, model, imagenetA_masking, MC=MC)
                        cumulative_accuracy += int(target == y_pred)
                    end_time_prediction = time.time()
                    time_dict["get_prediction"] += (end_time_prediction - start_time_prediction)
            else:
                start_time_prediction = time.time()
                with torch.no_grad():
                    inputs = normalize_input(inputs)
                    y_pred, _ = self.get_prediction(inputs, model, imagenetA_masking, MC=MC)
                    correct_predictions = (targets == y_pred).sum().item()
                    cumulative_accuracy += correct_predictions
                end_time_prediction = time.time()
                time_dict["get_prediction"] += (end_time_prediction - start_time_prediction)

            samples += inputs.shape[0]

            if verbose and batch_idx % log_interval == 0:
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(test_loader)}, Accuracy: {current_accuracy:.2f}%', end='\r')

        accuracy = cumulative_accuracy / samples * 100
        time_dict["total_time"] += sum(time_dict.values())

        self.save_result(accuracy = accuracy,
                         path_result = path_result,
                         seed_augmentations = seed_augmentations,
                         num_augmentations = num_augmentations,
                         augmentations = augmentations,
                         top_augmentations = top_augmentations,
                         MEMO = MEMO,
                         num_adaptation_steps = num_adaptation_steps,
                         lr_setting = lr_setting,
                         weights = weights_name,
                         prior_strength = prior_strength,
                         use_MC = use_MC,
                         time_test = time_dict)

        return accuracy