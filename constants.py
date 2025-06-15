from torchvision import transforms

transforms_tr = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ], p=0.9),
     
    transforms.Resize((260,260)),
    transforms.RandomCrop((256,256)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
   )
])

transforms_test = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
   )
])

dataset_archive_root = "/content/drive/MyDrive/datasets/catsvdogs.zip"
dataset_root = "/content/train"
save_path = "/content/drive/MyDrive/dl_models/catsvdogs"

class_labels = [
    "Cute kitty",
    "Stinky puppy"
]

class_dir_names =[
    "cats",
    "dogs"
    
]