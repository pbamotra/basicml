---
layout: post
title: "Squeezing image loading performance using libjpeg in Pytorch"
categories: 
    - performance
tags: pytorch augmentation libjpeg-turbo python image-processing 
cover_art: url(https://ucarecdn.com/8ee127cb-8f26-4a92-ae87-59fdea7b4727/pugsqueeze.png) no-repeat right
cover_art_size: 100%
cover_attribution: pixabay.com
---
Sometimes in your project you seek cheap thrills out of changing couple of lines of code. This is what happened when I came across the [libjpeg-turbo](https://libjpeg-turbo.org/){:target="_blank"} project. The turbo version of libjpeg boasts to be 2-6x faster in performing image processing operations. Performance is what everyone wants, right?

I work with Pytorch and so I started looking for Python bindings for libjpeg-turbo. Fortunately, Github user [@ajkxyz](https://github.com/ajkxyz/){:target="_blank"} has provided cffi [bindings](https://github.com/ajkxyz/jpeg4py){:target="_blank"} for libjpeg-turbo. Great! As per the author, in single threaded mode, we should expect a 30% improvement in image loading operations using these bindings. I was all in. 👀

Using these Python bindings is pretty straight-forward, so I went ahead to implement an end-to-end Pytorch dataloader that uses libjpeg-turbo backend to load images. Further, I also numpy-fied the random horizontal and vertical flipping operations to avoid expensive PIL.Image calls. For the code below, I assume that the image folder is flat and contain images with same dimensions.

{% highlight python %}
import os
import fire
import glob

import torch
import numpy as np
from PIL import Image
import jpeg4py as jpeg
from torch.utils import data


class MyDataSet(data.Dataset):
    """
    A custom dataset to load images
    """
    def __init__(self, *args, **kwargs):
        self.image_files = kwargs.get('image_files')
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        img = self.image_files[index]
        # -- PIL.Image loader -- #
        # return np.array(Image.open(img))
        # -- libjpeg-turbo loader -- #
        return jpeg.JPEG(img).decode()


def fast_collate(images) -> torch.Tensor:
    """
    Converts input batches from numpy arrays to Pytorch tensors
    :param images: batch consisting of images
    :return: images as tensors
    """
    
    # -- implements random horizontal and vertical flipping -- #
    images = [np.flip(img, axis=1) if np.random.rand() > 0.5 else img for img in images]
    images = [np.flip(img, axis=0) if np.random.rand() > 0.5 else img for img in images]

    w, h = images[0].shape[:2]
    image_tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8)
    for i, image_array in enumerate(images):
        if image_array.ndim < 3:
            image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.rollaxis(image_array, 2)
        # -- np.ascontiguousarray is needed becase np.flip operation stores negative strides -- #
        image_tensor[i] += torch.from_numpy(np.ascontiguousarray(image_array))
    return image_tensor

def run(image_folder: str):
    """
    Iterates over one single batch of images
    :param image_folder: flattened folder contained jpeg images
    """
    assert os.path.exists(image_folder), f"{image_folder} does not exist"
    files = glob.glob(image_folder + "*.jpg")
    data_set = MyDataSet(image_files=files)
    dl = iter(data.DataLoader(data_set, batch_size=32, num_workers=4, 
                              pin_memory=True, collate_fn=fast_collate))
    next(dl)

if __name__ == "__main__":
    fire.Fire(run)
{% endhighlight %}

I use a custom collate function because the in-built toTensor is a pretty involved function which slows down the whole array to tensor creation part. This collate function has been borrowed from the [Nvidia/apex](https://github.com/NVIDIA/apex){:target="_blank"} which lets you do half-precision training. Let's keep that discussion for another post.

[![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/drive/1WUvqftk1d7t7p2aGS-_Jaf_eK1170ZhZ)

I was able to squeeze +20% performance. Not bad! Change the batch size for more juice maybe?

Stay classy.

#### Links:
 - libjpeg-turbo [project](https://libjpeg-turbo.org/)
 - cffi Python bindings [repo](https://github.com/ajkxyz/jpeg4py) for libjpeg-turbo
