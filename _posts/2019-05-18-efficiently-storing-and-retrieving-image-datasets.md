---
layout: post
title: "Efficiently processing large image datasets in Python"
categories: 
    - performance
tags: python pytorch lmdb images computer-vision deep-learning
cover_art: url(/_assets/imgs/icons8/flamenco/flamenco-done.png) no-repeat right
cover_art_size: 100%
cover_attribution: icons8.com/ouch
---
<!-- <script type="module" src="/_assets/js/2019-05-18.js"></script> -->
I have been working on Computer Vision projects for some time now and moving from NLP domain the first thing I realized was that image datasets are yuge! I typically process 500GiB to 1TB of data at a time while training deep learning models. Out of the box, I rely on using `ImageFolder` class of Pytorch but disk reads are so slow (innit?). I was reading through open source projects <!--break--> to see how people efficiently process large image data sets like [Places](https://places2.csail.mit.edu/download.html){:target="_blank"}. That's how I stumbled into [LMDB](https://symas.com/lmdb/){:target="_blank"} store which is the focus of this post. The tagline on the official project page justifies the benefits of using LMDB: -

> An ultra-fast, ultra-compact, crash-proof key-value embedded data store.

In simple words, we will store images as key value pairs where keys are uniquely identifiable IDs for each image and values are numpy arrays stored as bytes and additional image related metadata. Let's see how an image folder can be processed and converted to an LMDB store.

```python
# lmdbconverter.py

import os
import cv2
import fire
import glob
import lmdb
import logging
import pyarrow
import lz4framed
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import jpeg4py as jpeg
from itertools import tee
from typing import Generator, Any


logging.basicConfig(level=logging.INFO,
                    format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
DATA_DIRECTORY = './data'
IMAGE_NAMES_FILE = 'image_names.csv'


def list_files_in_folder(folder_path: str) -> Generator:
    return (file_name__str for file_name__str in glob.glob(os.path.join(folder_path, "*.*")))


def read_image_safely(image_file_name: str) -> np.array:
    try:
        return jpeg.JPEG(image_file_name).decode().astype(np.uint8)
    except jpeg.JPEGRuntimeError:
        return np.array([], dtype=np.uint8)


def serialize_and_compress(obj: Any):
    return lz4framed.compress(pyarrow.serialize(obj).to_buffer())


def extract_image_name(image_path: str) -> str:
    return image_path.split('/').pop(-1)


def resize(image_array, size=(256, 256)):
    if image_array.size == 0:
        return image_array
    return cv2.resize(image_array, dsize=size, interpolation=cv2.INTER_CUBIC)


def convert(image_folder: str, lmdb_output_path: str, write_freq: int=5000):
    assert os.path.isdir(image_folder), f"Image folder '{image_folder}' does not exist"
    assert not os.path.isfile(lmdb_output_path), f"LMDB store '{lmdb_output_path} already exists"
    assert not os.path.isdir(lmdb_output_path), f"LMDB store name should a file, found directory: {lmdb_output_path}"
    assert write_freq > 0, f"Write frequency should be a positive number, found {write_freq}"

    logger.info(f"Creating LMDB store: {lmdb_output_path}")

    image_file: Generator = list_files_in_folder(image_folder)
    image_file, image_file__iter_c1, image_file__iter_c2, image_file__iter_c3 = tee(image_file, 4)

    img_path_img_array__tuples = map(lambda tup: (tup[0], resize(read_image_safely(tup[1]))),
                                     zip(image_file__iter_c1, image_file__iter_c2))

    lmdb_connection = lmdb.open(lmdb_output_path, subdir=False,
                                map_size=int(4e11), readonly=False,
                                meminit=False, map_async=True)

    lmdb_txn = lmdb_connection.begin(write=True)
    total_records = 0

    try:
        for idx, (img_path, img_arr) in enumerate(tqdm(img_path_img_array__tuples)):
            img_idx: bytes = u"{}".format(idx).encode('ascii')
            img_name: str = extract_image_name(image_path=img_path)
            img_name: bytes = u"{}".format(img_name).encode('ascii')
            if idx < 5:
                logger.debug(img_idx, img_name, img_arr.size, img_arr.shape)
            lmdb_txn.put(img_idx, serialize_and_compress((img_name, img_arr.tobytes(), img_arr.shape)))
            total_records += 1
            if idx % write_freq == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_connection.begin(write=True)
    except TypeError:
        logger.error(traceback.format_exc())
        lmdb_connection.close()
        raise

    lmdb_txn.commit()

    logger.info("Finished writing image data. Total records: {}".format(total_records))

    logger.info("Writing store metadata")
    image_keys__list = [u'{}'.format(k).encode('ascii') for k in range(total_records)]
    with lmdb_connection.begin(write=True) as lmdb_txn:
        lmdb_txn.put(b'__keys__', serialize_and_compress(image_keys__list))

    logger.info("Flushing data buffers to disk")
    lmdb_connection.sync()
    lmdb_connection.close()

    # -- store the order in which files were inserted into LMDB store -- #
    pd.Series(image_file__iter_c3).apply(extract_image_name).to_csv(os.path.join(DATA_DIRECTORY, IMAGE_NAMES_FILE),
                                                                    index=False, header=False)
    logger.info("Finished creating LMDB store")


if __name__ == '__main__':
    fire.Fire(convert)
```

To convert a flattened image folder to LMDB store, just run the following command.

```bash
python3 lmdbconverter.py --image_folder ./images/ --lmdb_output_path ./data/lmdb-store.db
```

There are a couple of things to notice in the code above. We use lz4 compression (fastest that I know of) to store our images in the LMDB store. This greatly reduces the size of output .db file produced. One thing to keep in mind is that to create an LMDB store one needs to know beforehand the expected size the .db file is going to need. I've resized all the images to 256x256x3 which is also needed for practical applications. That way I can calculate the required db size as 1600x256x256x3 bytes. I run the benchmark on ~1600 images ranging from 300x300 to 1000x1000 resolution. It takes less than 7 seconds to create the database and occupies merely 10% more space than images on the disk. Next, we create a Pytorch dataloader to read this LMDB store.

```python
# lmdbloader.py

import os
import lmdb
import pyarrow
import lz4framed
import numpy as np
from typing import Any
import nonechucks as nc
from torch.utils.data import Dataset, DataLoader


class InvalidFileException(Exception):
    pass


class LMDBDataset(Dataset):
    def __init__(self, lmdb_store_path, transform=None):
        super().__init__()
        assert os.path.isfile(lmdb_store_path), f"LMDB store '{lmdb_store_path} does not exist"
        assert not os.path.isdir(lmdb_store_path), f"LMDB store name should a file, found directory: {lmdb_store_path}"

        self.lmdb_store_path = lmdb_store_path
        self.lmdb_connection = lmdb.open(lmdb_store_path,
                                         subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb_connection.begin(write=False) as lmdb_txn:
            self.length = lmdb_txn.stat()['entries'] - 1
            self.keys = pyarrow.deserialize(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            print(f"Total records: {len(self.keys), self.length}")
        self.transform = transform

    def __getitem__(self, index):
        lmdb_value = None
        with self.lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[index])
        assert lmdb_value is not None, f"Read empty record for key: {self.keys[index]}"

        img_name, img_arr, img_shape = LMDBDataset.decompress_and_deserialize(lmdb_value=lmdb_value)
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        if image.size == 0:
            raise InvalidFileException("Invalid file found, skipping")
        return image

    @staticmethod
    def decompress_and_deserialize(lmdb_value: Any):
        return pyarrow.deserialize(lz4framed.decompress(lmdb_value))

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = nc.SafeDataset(LMDBDataset('./data/lmdb-tmp.db'))
    batch_size = 64
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=False)
    n_epochs = 50

    for _ in range(n_epochs):
        for batch in data_loader:
            assert len(batch) > 0
```

To run the dataloader, simply execute the script above. I've used [nonechucks](https://github.com/msamogh/nonechucks){:target="_blank"}, which basically removes bad images from a batch. Simple and neat, isn't it? According to my tests, I was able to achieve 65% reduction in time iterating over data set! That's good amount savings in terms of training time. I'm currently working on integrating this code to store all of my training image data set. Let's see how that goes.

[MIT License](/_assets/license.txt){:target="_blank"}

Happy coding. Stay classy.

#### Links:
 - Places [dataset](https://places2.csail.mit.edu/download.html)
 - LMDB [project](https://symas.com/lmdb/)
 - nonechucks [project](https://github.com/msamogh/nonechucks)
