# -*- coding:utf-8 -*-

import gdown
import os


if __name__ == "__main__":
    url = "https://drive.google.com/uc?export=download&id=1HRyVO1qGm29sTS9xzPRuKcFd-A4SB5BU"
    output = "data_volume.zip"
    gdown.cached_download(url, output, postprocess=gdown.extractall)
    os.remove("data_volume.zip")
