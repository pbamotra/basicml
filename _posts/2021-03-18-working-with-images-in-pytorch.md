---
layout: post
title: "Working with broken images in Pytorch"
categories: 
    - performance
tags: python pytorch images 
cover_art: url(/_assets/imgs/icons8/cherry/cherry-fatal-error.png) no-repeat right
cover_art_size: 80%
cover_attribution: icons8.com/ouch
---
Too often I've found myself in this problem with Pytorch where the [dataloader](https://pytorch.org/docs/stable/data.html){:target="_blank"} doesn't work because there's a bad image in the dataset. One solution would definitely be to write a module that loads each image and then deletes the bad ones. But, I wanted something elegant and the following code is an attempt at smoothly ignoring the bad images in batches while also being able to process non-RGB images.
<!--break-->

```python
# torchimageprocessor.ipynb

# Install Pillow-SIMD - https://github.com/uploadcare/pillow-simd
!pip uninstall pillow
!CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Download a sample image dataset - https://www.robots.ox.ac.uk/~vgg/data/pets/
!rm -rf images/ images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz && tar -xzf images.tar.gz
!mkdir ./images/0
!mv ./images/*.jpg ./images/0

# Add a sample invalid image -- Too small dimension
# wget https://raw.githubusercontent.com/mathiasbynens/small/master/jpeg.jpg 
# cp jpeg.jpg ./images/0


import torch
from PIL import Image
from torchvision import datasets, transforms

MIN_VALID_IMG_DIM = 100
IMG_CROP_SIZE = 224


def rgb_loader(path):
    img = Image.open(path)
    if img.getbands() != ('R', 'G', 'B'):
        img = img.convert('RGB')
    return img

def is_valid_file(path):
    try:
        img = Image.open(path)
        img.verify()
    except:
        return False
    
    if not(img.height >= MIN_VALID_IMG_DIM and img.width >= MIN_VALID_IMG_DIM):
        return False

    return True


train_transformations = transforms.Compose([transforms.RandomResizedCrop(IMG_CROP_SIZE),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

data = datasets.ImageFolder(root='./images', 
                            loader=rgb_loader, 
                            is_valid_file=is_valid_file, 
                            transform=train_transformations)

num_processors = !nproc
num_workers = max(64, int(num_processors[0]))

dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, drop_last=False, num_workers=num_workers)

total_img_files = !ls -f1 ./images/0 | wc -l
total_img_files = int(total_img_files[0]) - 2

imgs_seen = 0
for imgs, labels in dataloader:
    assert len(imgs) > 0, "Bad batch formed"
    imgs_seen += len(imgs)

print(f"Total images on disk: {total_img_files}")
print(f"Total images seen: {imgs_seen}")
```

I've mentioned about [nonechucks](https://github.com/msamogh/nonechucks){:target="_blank"} in one of my previous posts [here](/performance/2019/05/18/efficiently-storing-and-retrieving-image-datasets.html). But, the solution presented above is using native Pytorch API and looks much simpler.

[![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/drive/1362I-QueZ4kN0zbsKIrfnhNCvj_dnrYJ?usp=sharing)
[MIT License](/_assets/license.txt){:target="_blank"}

Happy coding. Stay classy.

#### Links:
 - Pets [dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
 - nonechucks [project](https://github.com/msamogh/nonechucks)
