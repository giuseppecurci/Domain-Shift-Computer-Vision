import torch
import torchvision.transforms as T
import random 

def compute_entropy(probabilities):
    """
    Takes a tensor of probabilities [1,Classes] and computes the entropy returned as one-dimensional tensor.
    """
    # Ensure probabilities are normalized (sum to 1)
    if not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
        raise ValueError("The probabilities should sum to 1.")

    # Compute entropy
    # Adding a small value to avoid log(0) issues
    epsilon = 1e-10
    probabilities = torch.clamp(probabilities, min=epsilon)
    entropy = -torch.sum(probabilities * torch.log(probabilities))

    return entropy

def get_best_augmentations(probabilities, top_k):
    """
    Takes a tensor of probabilities with dimension [num_augmentations,classes] or [mc_models,num_augmentations,200] 
    and outputs a tensor containing the probabilities corresponding to the augmentations 
    with the lowest entropy of dimension [top_k, classes] or [mc_models, top_k, classes].
    ----------
    top_k: number of augmentations to select
    probabilities: a tensor of dimension [num_augmentations,200] 
    """
    if probabilities.dim() == 2:
        probabilities = probabilities.unsqueeze(0)

    # nested list comprehension needed if probabilities is a 3D tensor (MC dropout)
    entropies = torch.tensor([[compute_entropy(prob) for prob in prob_set] for prob_set in probabilities])
    _, top_k_indices = torch.topk(entropies, top_k, largest=False, sorted=False)
    sorted_top_k_indices = torch.stack([indices[torch.argsort(entropies[i, indices])] 
                                            for i, indices in enumerate(top_k_indices)])
    top_k_probabilities = torch.stack([probabilities[i][sorted_top_k_indices[i]] 
                                        for i in range(probabilities.shape[0])])
    if top_k_probabilities.shape[0] == 1:
        top_k_probabilities = top_k_probabilities.squeeze(0)

    return top_k_probabilities

def get_test_augmentations(input, augmentations, num_augmentations, seed_augmentations):
    """
    Takes a tensor image of dimension [C,H,W] and returns a tensor of augmentations of dimension [num_augmentations, C,H,W]. 
    The augmentations are produced by sampling different torchvision.transforms from "augmentations". 
    ----------
    input: an image tensor of dimension [C,H,W]
    augmentations: a list of torchvision.transforms augmentations
    num_augmentations: the number of augmentations to produce
    seed_augmentations: seed to reproduce the sampling of augmentations
    """
    torch.manual_seed(seed_augmentations)
    random.seed(seed_augmentations)
    sampled_augmentations = random.sample(augmentations, num_augmentations)
    test_augmentations = torch.zeros((num_augmentations, 3, 224, 224))
    for i, augmentation in enumerate(sampled_augmentations):
        transform_MEMO = T.Compose([
            T.ToPILImage(),
            augmentation,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        augmented_input = transform_MEMO(input.cpu())
        test_augmentations[i] = augmented_input
    return test_augmentations