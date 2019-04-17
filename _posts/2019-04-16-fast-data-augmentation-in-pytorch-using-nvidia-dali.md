---
layout: post
title: "Fast data augmentation in Pytorch using Nvidia DALI"
categories: 
    - performance
tags: pytorch augmentation gpu-processing nvidia DALI
cover_art: url(/assets/imgs/icons8/cherry/cherry-message-sent.png) no-repeat right
cover_art_size: 90%
cover_attribution: icons8.com/ouch
---
In my new project at work I had to process a sufficiently large set of image data for a multi-label multi-class classification task. Despite the GPU utilization being close to 100%, a single training epoch over 2 million images took close to 3.5 hrs to run. This is a big issue if you're running your baseline experiments and want quick results. I first thought that since I was processing original size images each of which were at least a few MBs the bottleneck was disk I/O. I used [Imagemagick mogrify](https://imagemagick.org/script/mogrify.php){:target="_blank"} to resize all 2 million images which took a long time. To my astonishment resizing images didn't reduce the training time at all! Well, not noticeably. So, I went through the code and found out that the major bottleneck were the image augmentation operations in Pytorch. 

{% highlight python %}
from torchvision import transforms

def get_image_transforms() -> transforms.Compose:
    """
    These transformations meant for data augmentation are a bottleneck
    since all the operations are done on CPU and then the tensors are
    copied to the GPU device.
    """
    return transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
{% endhighlight %}

While stumbling on github I found that people working at Nvidia had recently released a library - [DALI](https://github.com/NVIDIA/DALI) that is supposed to tackle exactly this issue. The library is still under active development and supports fast data augmentation for all major ML development libraries out there - [Pytorch](https://pytorch.org/), [Tensorflow](https://www.tensorflow.org/), [MXNet](https://mxnet.apache.org/).

<figure>
{% svg /assets/imgs/data-pipeline.svg alt="Typical data pipeline" %}
<figcaption style="text-align: center">Fig 1: A typical data augmentation pipeline</figcaption>
</figure>

Using Nvidia DALI, the above data pipeline can be optimized by moving appropriate [operations](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/supported_ops.html){:target="_blank"} to GPU. After using DALI, the pipeline looks something like -

<figure>
{% svg /assets/imgs/dali-pipeline.svg alt="Nvidia DALI pipeline" %}
<figcaption style="text-align: center">Fig 2: An Nvidia DALI pipeline</figcaption>
</figure>

For more details about features of DALI, please see this beginner friendly post by Nvidia developers titled <i>[Fast AI Data Preprocessing with NVIDIA DALI](https://devblogs.nvidia.com/fast-ai-data-preprocessing-with-nvidia-dali/){:target="_blank"}</i>. In the rest of this post, I'll show how to incorporate Nvidia DALI in your Pytorch code. The readers are welcome to offer possible improvements to the code below.

We start by installing the required dependencies.

{% highlight bash %}
# Find out the cuda version so that we install appropriate DALI binaries
# Find installation instructions at 
# https://github.com/NVIDIA/DALI#installing-prebuilt-dali-packages

$ nvcc --version

# sample output
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
{% endhighlight %}

{% highlight bash %}
# install Nvidia DALI python bindings
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

# sample output
Looking in indexes: https://pypi.org/simple, https://developer.download.nvidia.com/compute/redist/cuda/10.0
Collecting nvidia-dali
  Downloading https://developer.download.nvidia.com/compute/redist/cuda/10.0/nvidia-dali/nvidia_dali-0.8.1-699137-cp36-cp36m-manylinux1_x86_64.whl (30.0MB)
    100% |████████████████████████████████| 30.0MB 1.3MB/s 
Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from nvidia-dali) (0.16.0)
Installing collected packages: nvidia-dali
Successfully installed nvidia-dali-0.8.1 
{% endhighlight%}

By now you have completed installation of `nvidia-dali` that we'll now integrate into our Pytorch code. To create a dummy data set, we download the flower classification data provided by [Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188). The dataset contains two folders - `train` and `valid`. We use the images in the `train` folder and flatten the directory which comes organized as a hierarchical folder containing images by label with one sub-folder per label. We don't use the provided labels and generate dummy labels for demonstration.

{% highlight bash %}
$ wget -cq https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip \
  && unzip -qq flower_data.zip \
  && mkdir -p ./flower_data/flower_data_flat \
  && find ./flower_data/train -mindepth 2 -type f -exec mv -t ./flower_data/flower_data_flat -i '{}' +
{% endhighlight %}

Next we create a space separated file that fits the example given on official Nvidia DALI documentation pages.

{% highlight python %}
from os import listdir
from os.path import isfile, join

images_directory = './flower_data/flower_data_flat'
# read names of all image files
image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

# we create a data frame with the image names and dummy labels - label_1, label_2
data = pd.DataFrame(list(zip(image_files, 
                            list(range(len(image_files))), 
                            list(range(len(image_files))))), 
                    columns=['image_filename', 'label_1', 'label_2'])

processed_data_file = 'flower_dummy_data.ssv'
data.to_csv(processed_data_file, index=False, header=False, sep=' ')

print(data.head())
#	image_filename	label_1	label_2
# 0	image_05973.jpg	0	    0
# 1	image_00956.jpg	1	    1
# 2	image_06047.jpg	2	    2
# 3	image_07168.jpg	3	    3
# 4	image_04466.jpg	4	    4
{% endhighlight %}

Next we create an `ExternalInputIterator` that batches our data and is used by DALI [Pipeline](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/examples/getting%20started.html#Pipeline){:target="_blank"} to input the data and feed it to respective devices for processing. The code below has been adapted from the official code [here](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/external_input.html){:target="_blank"} to work for multiple labels. Thanks to [@Siddha Ganju](https://twitter.com/SiddhaGanju){:target="_blank"} for pointing to the official tutorial.

{% highlight python %}
import types
import numpy as np
import collections
import pandas as pd

from random import shuffle

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

class ExternalInputIterator(object):
    def __init__(self, batch_size, data_file, image_dir):
        self.images_dir = image_dir
        self.batch_size = batch_size
        self.data_file = data_file
        with open(self.data_file, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            # *label reads multiple labels 
            jpeg_filename, *label = self.files[self.i].split(' ')
            f = open(image_dir + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array(label, dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__
{% endhighlight %}

Next we instantiate this iterator and feed it as an input to `ExternalSourcePipeline` that extends the `Pipeline` class and feeds data to respective devices for augmentation operations.

{% highlight python %}
eii = ExternalInputIterator(batch_size=16, 
                            data_file=processed_data_file, 
                            image_dir=images_directory)
iterator = iter(eii)

class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.data_iterator = data_iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        # resizing is *must* because loaded images maybe of different sizes
        # and to create GPU tensors we need image arrays to be of same size
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        output = self.res(images)
        return (output, self.labels)

    def iter_setup(self):
        # the external data iterator is consumed here and fed as input to Pipeline
        images, labels = self.data_iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)
{% endhighlight%}

We are almost done and now we instantiate a `DALIGenericIterator` that helps us iterate over the dataset just the way we do typically in Pytorch.

{% highlight python %}
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = ExternalSourcePipeline(batch_size=16, num_threads=2, device_id=0)
pipe.build()

# first parameter is list of pipelines to run
# second pipeline is output_map that maps consecutive outputs to 
#   corresponding names
# last parameter is the number of iterations - number of examples you
# want to iterate on
dali_iter = DALIGenericIterator([pipe], ['images', 'labels'], 256)

for i, it in enumerate(dali_iter):
    batch_data = it[0]
    images, labels = batch_data["images"], batch_data["labels"]
    # both images and labels are `torch.Tensor` which can now be processed
    # the way we usually do in Pytorch example -https://urlzs.com/Wa2b
    
    # the rest of the code in this block looks something like
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
{% endhighlight %}

<i class="fab fa-google"></i> Colab notebook: [http://tiny.cc/nvidia-dali](http://tiny.cc/nvidia-dali)

I'm yet to benchmark DALI in my code and will update this post once I've the results. 

Stay classy.

#### Links:
 - DALI Github [repo](https://github.com/NVIDIA/DALI)
 - Official [blog post](https://devblogs.nvidia.com/fast-ai-data-preprocessing-with-nvidia-dali/) on DALI
