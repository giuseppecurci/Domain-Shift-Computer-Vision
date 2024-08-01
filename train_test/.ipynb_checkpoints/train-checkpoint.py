import os
# to import modules from other directories
os.chdir("/home/sagemaker-user/Domain-Shift-Computer-Vision") 
print("Warning: the working directory was changed to", os.getcwd())

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from train_test.test import Tester

class Trainer(Tester):

    def __init__(self, 
                 data_loaders: dict, 
                 dataset_name: str,
                 model: torch.nn.Module, 
                 optimizer: torch.optim, 
                 loss_fn: torch.nn, 
                 device, 
                 seed: int,
                 exp_name, # the name of this experiment
                 exp_path, # where you keep all the experiments
                 use_early_stopping=True,
                 patience=5,
                 delta=1e-3,
                 scheduler=None,
                 num_classes = None,
                 trained = False):
        """
        The exp_name should be a string containing all the information about the experiment:
        - model 
        - optimizer 
        - loss function
        - other hyperparameters 
        """
        super().__init__(model, data_loaders, device, loss_fn, num_classes)
        self.__optimizer = optimizer        
        self.__use_early_stopping = use_early_stopping
        self.__seed = seed
        self.__epoch = 0
        self.__trained = trained
        self.__scheduler = scheduler
        self.__best_loss = float("inf")

        assert os.path.exists(exp_path), "Experiment path does not exist"
        assert os.path.exists(os.path.join(exp_path, exp_name)) == False, "The experiment already exists"
        assert isinstance(patience, int) and patience > 0, "patience must be a positive integer"
        assert isinstance(delta, float) and delta > 0, "delta must be a positive float"
        assert isinstance(use_early_stopping, bool), "use_early_stopping must be a boolean"
        assert isinstance(data_loaders, dict), "data_loaders must be a dictionary with keys 'train_loader', 'val_loader', 'test_loader'"

        self.__exp_name = os.path.join(exp_path, exp_name)     

        if self.__use_early_stopping:
            from utility.early_stopping import EarlyStopping
            self.__early_stopping = EarlyStopping(patience=patience, 
                                                  delta=delta,
                                                  path=os.path.join(self.__exp_name,"checkpoint.pth"))      

        os.makedirs(self.__exp_name, exist_ok=True) 
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        self.__save_config(dataset_name, patience, delta)

    def get_model(self):
        return self._Tester__model
    
    def get_optimizer(self):
        return self.__optimizer
    
    def get_scheduler(self):
        return self.__scheduler
    
    def get_device(self):
        return self._Tester__device
    
    def get_exp_name(self):
        return self.__exp_name

    def __train_step(self, verbose, log_interval):
        """
        log_interval can be an integer or a float between 0 and 1. If it is an integer, the function will print the statistics every log_interval steps.
        If it is a float, the function will print the statistics every log_interval*num_of_batches steps.
        """
        samples = 0.0
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0
        
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert isinstance(log_interval, (int, float)) and log_interval>0, "log_interval must be an integer or a float and non-negative"
        if log_interval < 1:
            log_interval = int(len(self._Tester__data_loaders["train_loader"])*log_interval)

        self._Tester__model.train()

        for batch_idx, (inputs, targets) in enumerate(self._Tester__data_loaders["train_loader"]):
            inputs, targets = inputs.to(self._Tester__device), targets.to(self._Tester__device)
            
            outputs = self._Tester__model(inputs)
            loss = self._Tester__loss_fn(outputs, targets)
            loss.backward()
            self.__optimizer.step()
            self.__optimizer.zero_grad()
            cumulative_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

            samples += inputs.shape[0]

            if verbose and batch_idx % log_interval == 0:
                current_loss = cumulative_loss / samples
                current_accuracy = cumulative_accuracy / samples * 100
                print(f'Batch {batch_idx}/{len(self._Tester__data_loaders["train_loader"])}, Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.2f}%', end='\r')
            
        if self.__scheduler:
            self.__scheduler.step()

        accuracy = cumulative_accuracy / samples * 100    
        avg_loss = cumulative_loss / samples    

        return avg_loss, accuracy

    def main(self,
             epochs=10,
             verbose_steps=True, # print after log_interval-learning steps
             log_interval=10): 

        from utility.initialize import initialize
        initialize(self.__seed)
            
        self._Tester__model.to(self._Tester__device)
                
        # Log to TensorBoard
        if self.__trained == False:
            self.__trained = True
            print("Before training:")
            train_loss, train_accuracy = self.test_step(train=True)
            val_loss, val_accuracy = self.test_step(eval=True) 
            test_loss, test_accuracy = self.test_step(test=True)
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")
            self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, "Test")
            self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)
        
        pbar = tqdm(range(epochs), desc="Training")
        for _ in pbar:
            train_loss, train_accuracy = self.__train_step(verbose=verbose_steps, log_interval=log_interval)
            val_loss, val_accuracy = self.test_step(eval=True) 

            print("-----------------------------------------------------")
            self.__epoch += 1
            self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
            self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")

            pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)

            if self.__use_early_stopping:
                self.__early_stopping(val_loss,self._Tester__model, self.__optimizer, self.__scheduler)
                if self.__early_stopping.early_stop:
                    print("Early stopping")
                    break       
            else:
                if val_loss < self.__best_loss:
                    self.__best_loss = val_loss
                    torch.save({
                        "model": self._Tester__model.state_dict(),
                        "optimizer": self.__optimizer.state_dict(),
                        "scheduler": self.__scheduler.state_dict() if self.__scheduler is not None else None
                        }, 
                        os.path.join(self.__exp_name,"checkpoint.pth"))
        
        # Compute final evaluation results
        print("After training:")
        train_loss, train_accuracy = self.test_step(train=True)
        val_loss, val_accuracy = self.test_step(eval=True) 
        test_loss, test_accuracy = self.test_step(test=True)

        self.__log_values(self.__writer, self.__epoch, train_loss, train_accuracy, "Train")
        self.__log_values(self.__writer, self.__epoch, val_loss, val_accuracy, "Validation")
        self.__log_values(self.__writer, self.__epoch, test_loss, test_accuracy, "Test")

        self.__print_statistics(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)

        # Flush the logs to disk 
        self.__writer.flush()            

    def close_writer(self):
        self.__writer.close()
        print("Writer closed")

    def open_writer(self):
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        print("A new writer was opened ")

    def set_exp_name(self, new_name):
        self.__exp_name = new_name
        self.__writer = SummaryWriter(log_dir=f"{self.__exp_name}")
        print(f"Experiment name was changed to {new_name}")
    
    def __print_statistics(self, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
        print("-----------------------------------------------------")

    # tensorboard logging utilities
    def __log_values(self, writer, step, loss, accuracy, prefix):
        writer.add_scalar(f"{prefix}/loss", loss, step)
        writer.add_scalar(f"{prefix}/accuracy", accuracy, step)
        
    def __save_config(self, dataset_name, patience, delta):
        config = {
            'data': { 
                'batch_size':self._Tester__data_loaders["train_loader"].batch_size,
                'dataset_name': dataset_name
            }, 
            'model': self._Tester__model.__class__.__name__,
            'optimizer': {
                'optimizer': self.__optimizer.__class__.__name__,
                'momentum': self.__optimizer.param_groups[0]['momentum'],
                'weight_decay': self.__optimizer.param_groups[0]['weight_decay'],
                'lr': self.__optimizer.param_groups[0]['lr'],
            } ,
            'scheduler': self.__scheduler.__class__.__name__ if self.__scheduler is not None else None,
            'loss_fn': {
                'loss_fn': self._Tester__loss_fn.__class__.__name__,
                'smoothing': self._Tester__loss_fn.label_smoothing
            },
            'seed': self.__seed,
            'early_stopping': {
                'use_early_stopping': self.__use_early_stopping,
                'patience': patience,
                'delta': delta
            }
        }
        import json
        config_file_path = f"{self.__exp_name}/config.json"
        with open(config_file_path, 'w') as file:
            json.dump(config, file, indent=4)