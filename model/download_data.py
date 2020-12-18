from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
from pathlib import Path
import argparse

from config.config import Config

Config.init_config()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='url to zipped dataset')
    parser.add_argument('--password', help='password zipped dataset')

    args = parser.parse_args()

    zipurl = args.url
    zippassword = args.password
    if zipurl is None:
        zipurl = Config.get_value("data", "url")
        zippassword = Config.get_value("data", "password")
    
    output_path = Config.get_value("model", "input", "path")
    # create dictionary
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # download the pretrained wordvec
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(output_path, None, zippassword.encode())