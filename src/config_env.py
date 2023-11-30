import os
from configparser import ConfigParser

def load_ini_env(filename):
    config = ConfigParser()
    config.read(filename)
    for key, value in config['environment'].items():
        os.environ[key] = value