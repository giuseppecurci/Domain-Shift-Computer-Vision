import torch 

def adaptive_bn_forward(self, input: torch.Tensor):
    """
    Applies an adaptive batch normalization to the input tensor using precomputed running 
    statistics that are updated in an adaptive manner using Schneider et al. [40] formula:
                        mean = N/(N+1)*mean_train + 1/(N+1)*mean_test
                        var = N/(N+1)*var_train + 1/(N+1)*var_test
    N corresponds to the weight that is given to the statistics of the pre-trained model.
    In the implementation, N/(N+1) corresponds to self.prior_strength and can be modified using
    nn.BatchNorm2d.prior_strength. 
    -----------
    input : input tensor of shape [N, C, H, W]
    """
    # compute channel-wise statistics for the input
    point_mean = input.mean([0,2,3]).to(device = self.running_mean.device) 
    point_var = input.var([0,2,3], unbiased=True).to(device = self.running_mean.device)
    # BN adaptation
    adapted_running_mean = self.prior * self.running_mean + (1 - self.prior_strength) * point_mean
    adapted_running_var = self.prior * self.running_var + (1 - self.prior_strength) * point_var
    # detach to avoid non-differentiable torch error
    adapted_running_mean = adapted_running_mean.detach()
    adapted_running_var = adapted_running_var.detach()
    
    return torch.nn.functional.batch_norm(input, adapted_running_mean, adapted_running_var, self.weight, self.bias, False, 0, self.eps)