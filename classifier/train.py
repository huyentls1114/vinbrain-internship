import argparse
import yaml
import torch.nn as nn


def train(configs):
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default = "./config/train_config.yaml", metavar = "FILE", type = str)
    args = parser.parse_args()
    
    #extract config
    config_file = open(args.config_file, 'r')
    configs = yaml.load(config_file)



