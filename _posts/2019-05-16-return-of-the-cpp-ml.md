---
layout: post
title: "C++ for ML dev - Flashlight / Arrayfire"
categories: 
    - vices
tags: c++ flashlight arrayfire machine-learning
cover_art: url(/assets/imgs/icons8/cherry/cherry-success.png) no-repeat right
cover_art_size: 100%
cover_attribution: icons8.com/ouch
---
My heart and soul belongs to Python, though at times I miss the performant code I once used to write in C++. Libraries like Pytorch abstract the fact that Python is way slower than C++ (sometimes 100x slower). While you can write Cython, I don't like taking away simplicity that Python offers. So, I like to keep my C++ and Python hats separate.

I've been coding exclusively in Python for almost 5 years now but the nostalgia of the C++ days just kicks in once in a while. I've started coding in C++ again, starting with some text processing 👀 (confession: sorry Python) and now working on unleashing the C++/CUDA two-headed beast. But, the problem is writing ML code in C++ is just a lot of boilerplate and I don't know CUDA programming. By lot, I mean a lot. And, that was the reason I got married to Python. Couple of `import` statements, model instantiation, and your MNIST classifier is running on multiple GPUs. 

This is when I stumbled across [flashlight](https://fl.readthedocs.io/en/latest/index.html){:target="_blank"}, an open-source C++ machine learning library by Facebook. In this post, I'll just describe how to begin with flashlight that finally leads us to a [hello-world](https://fl.readthedocs.io/en/latest/installation.html#building-your-project-with-flashlight){:target="_blank"} like example written in flashlight. For the installations purposes, I used a P3 instance on AWS running Ubuntu 16.04, CUDA 10.0, and CuDNN 7.5. I compiled flashlight with the CUDA backend only.

As pointed in the flashlight installation guide, we need to install [Arrayfire](https://arrayfire.com/download/){:target="_blank"} and [Cereal](https://github.com/USCiLab/cereal/){:target="_blank"} as a dependency first. So, let's start with that.

```bash
$ wget --no-check-certificate https://arrayfire.s3.amazonaws.com/3.6.3/ArrayFire-v3.6.3_Linux_x86_64.sh
# install Arrayfire in /opt location -- this is the recommended location
$ ./ArrayFire-v3.6.3_Linux_x86_64.sh --include-subdir --prefix=/opt

# install Cereal
$ git clone https://github.com/USCiLab/cereal.git
$ cd cereal && git checkout develop
$ mkdir build && cd build
$ cmake ..
$ make -j4
$ make install
```

Now we have dependencies installed, let's begin installation of flashlight.

```bash
$ git clone https://github.com/facebookresearch/flashlight.git
$ cd flashlight
$ mkdir build && cd build
$ ArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA
$ make -j4
$ make install
```

That completes the installation of everything we need. Now we can write our first code in flashlight. To do that let's create a cmake build configuration that can link flashlight to our project and helps to build the code seamlessly. Let's begin.

```bash
$ mkdir hello-flashlight
$ cd hello-flashlight
$ touch CMakeLists.txt
$ vi CMakeLists.txt
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ArrayFire REQUIRED)
find_package(flashlight REQUIRED)
add_executable(hello-fl hello-fl.cpp)
target_link_libraries(hello-fl flashlight::flashlight)
```

```bash
$ vi hello-fl.cpp
```

```cpp
#include <iostream>
#include <arrayfire.h>
#include "flashlight/flashlight.h"

int main() {
  fl::Variable v(af::constant(1, 1), true);
  auto result = v + 10;
  std::cout << "Hello World!" << std::endl;
  af::print("Array value is ", result.array());
  return 0;
}
```

```bash
$ pwd
/home/hello-flashlight

$ mkdir build && cd build
$ cmake ..
$ make
```

This generates the executable ```hello-fl``` inside the ```build``` directory. We can run this exectable.

```bash
$ ./hello-fl

Hello World!
Array value is 
[1 1 1 1]
   11.0000
```

That was a very simple example to begin with. The official documentation provides numerous examples to jumpstart with, like - [MNIST Classification](https://fl.readthedocs.io/en/latest/mnist.html){:target="_blank"}. I'm looking forward to experiment more and benchmark flashlight against Pytorch. Flashlight by design is extensible and one can write custom CUDA kernels too! I'm so eager to get my hands dirty with these extensions  to aquaint myself with CUDA programming.

Happy coding. Stay classy.

#### Links:
 - arrayfire [project](https://arrayfire.com/download/)
 - cereal [repo](https://github.com/USCiLab/cereal/)
 - flashlight [project](https://fl.readthedocs.io/en/latest/index.html)
 - flashligth [repo](https://github.com/facebookresearch/flashlight)
