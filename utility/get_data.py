from torch.utils.data import Dataset
import boto3 # read and write for AWS buckets
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch

class S3ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.s3_bucket = "deeplearning2024-datasets" # name of the bucket
        self.s3_region = "eu-west-1" # Ireland
        self.s3_client = boto3.client("s3", region_name=self.s3_region, verify=True)
        self.transform = transform

        # Get list of objects in the bucket
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=root)
        objects = response.get("Contents", [])
        while response.get("NextContinuationToken"):
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=root,
                ContinuationToken=response["NextContinuationToken"]
            )
            objects.extend(response.get("Contents", []))

        # Iterate and keep valid files only
        self.instances = []
        for ds_idx, item in enumerate(objects):
            key = item["Key"]
            path = Path(key)
            
            # Check if file is valid
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"):
                continue

            # Get label
            label = path.parent.name

            # Keep track of valid instances
            self.instances.append((label, key))

        # Sort classes in alphabetical order (as in ImageFolder)
        self.classes = sorted(set(label for label, _ in self.instances))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        try:
            label, key = self.instances[idx]
            
            # Download image from S3
            # response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            # img_bytes = response["Body"]._raw_stream.data

            img_bytes = BytesIO()
            response = self.s3_client.download_fileobj(Bucket=self.s3_bucket, Key=key, Fileobj=img_bytes) # download each image
            # img_bytes = response["Body"]._raw_stream.data
            
            # Open image with PIL
            img = Image.open(img_bytes).convert("RGB")

            # Apply transformations if any
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image at index {idx}: {str(e)}")

        return img, self.class_to_idx[label]

def get_data(batch_size, img_root, seed, split_data = True,transform = None):

    # Load data
    data = S3ImageFolder(root=img_root, transform=transform)

    if split_data:
        # Create train and test splits (80/20)
        num_samples = len(data)
        training_samples = int(num_samples * 0.8 + 1)
        val_samples = int(num_samples * 0.1)
        test_samples = num_samples - training_samples - val_samples
        
        torch.manual_seed(seed)
        training_data, val_data, test_data = torch.utils.data.random_split(data, [training_samples, val_samples, test_samples])
        
        # Initialize dataloaders
        train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader
        
    data_loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False, num_workers=4)    
    return data_loader