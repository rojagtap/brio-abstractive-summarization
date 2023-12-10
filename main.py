import shutil

import gdown
import nltk

from train import train

nltk.download('punkt')


if __name__ == "__main__":
    gdown.download(id="10MyeEZVSgh38ot3O9mEhPWPqxLSJMoxA")
    shutil.unpack_archive("cnndm.zip")

    train("cuda:0", "cnndm/diverse/train", "cnndm/diverse/val", "", "cnndm")
