import random 
import torch
import torchvision.transforms as T
from utility.get_data import get_data, S3ImageFolder

def MEMO(model, optimizer, device, augmentations, num_augmentations, seed, batch_size, img_root, checkpoint_path, loss, top_k = None):

    torch.manual_seed(seed)
    
    transform = T.Compose([
        T.Resize((256, 256)),  
        T.RandomCrop((224,224)),
        T.ToTensor()
    ])
    
    # loop over zipped data loaders with different transform
    _, _, test_loader = get_data(batch_size, img_root, transform = transform, seed = seed)

    sampled_augmentations = random.sample(augmentations, num_augmentations)
    checkpoint = torch.load(checkpoint_path)

    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        for input, target in zip(inputs, targets):
            model.load_state_dict(checkpoint["model"])
            model.train()
            model.to(device) 
            
            optimizer.load_state_dict(checkpoint["optimizer"])
            MEMO_augmentations = torch.zeros((num_augmentations, *input.shape), device=device)
            for i, augmentation in enumerate(sampled_augmentations):
                transform_2 = T.Compose([
                        augmentation,  
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                ])
                augmented_input = transform_2(input.cpu()).to(device)  # Apply the transformation and move to device
                MEMO_augmentations[i] = augmented_input
            
            outputs = model(MEMO_augmentations)
            marginal_output_distribution = torch.mean(outputs, dim=0)
            
            marginal_loss = loss(marginal_output_distribution, target)
            marginal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            input = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
            probab_pred = torch.softmax(model(input.unsqueeze(0)), dim=1)
            y_pred = probab_pred.argmax()
            cumulative_accuracy += int(target == y_pred)

        samples += inputs.shape[0]

    accuracy = cumulative_accuracy / samples * 100
    
    return accuracy

            