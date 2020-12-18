import os
from functools import reduce

import yaml


class Config:
    """ Get config from config file or environment variables.
    The priority order of config is: Environment > Config file.
    """

    config = dict()
    
    @staticmethod
    def init_config():
        with open("config/application.yaml", 'r') as ymlfile:
            Config.config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    @staticmethod
    def get_value(*keys):
        if os.environ.get('_'.join(keys).upper()):
            return os.environ.get('_'.join(keys).upper())
        else:
            return Config.get(*keys)

    @staticmethod
    def get_value_map(map):
        new_map = {}
        map_keys = Config.config.get(map).keys()
        for key in map_keys:
            new_map[key] = Config.get_value(map, key)
        return new_map

    @staticmethod
    def get(*keys):
        return reduce(lambda d, key: d.get(key) if d else None, keys, Config.config)
