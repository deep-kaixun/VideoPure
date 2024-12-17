import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_loaders(imgstr, batch_size=1, n_gpus=1):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    if imgstr == 'UCF101':
        testset= _ucf101('test')
    testloader = DataLoader(testset, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=4, prefetch_factor=2)

    return testloader


class ucf101_val(Dataset):
    def __init__(self,root,transform) -> None:
        super(ucf101_val).__init__()
        self.root=root
        self.transform=transform
        self.videos,self.labels=self.make_video(self.root)
        self.target_transform=None

    
    def make_video(self,root):
        videos,labels=[],[]
        for p in os.listdir(root):
            la=int(p)
            labels.append(la)
            p=os.path.join(root,p)
            videos.append(p)
        return videos,labels
            
                
    def loader(self,path: str):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.videos[index]
        target= self.labels[index]
        videos=[]
        for t in range(32):
            sample = self.loader(os.path.join(path,str(t)+'.png'))
            if self.transform is not None:
                sample = self.transform(sample)
            videos.append(sample)
        video=torch.stack(videos)
        video=video.permute(1,0,2,3)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return video, target
    def __len__(self) -> int:
        return len(self.videos)

        

def _ucf101(split: str) -> Dataset:
    dataset_path ='clean_videos'
    if split == "train":
        return datasets.ImageFolder(os.path.join(dataset_path,'train'), transform=transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        # print(dataset_path)
        return ucf101_val(root=dataset_path, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]))

    else:
        raise Exception("Unknown split name.")

