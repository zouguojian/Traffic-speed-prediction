# -- coding: utf-8 --
import argparse
from model.hyparameter import parameter
para = parameter(argparse.ArgumentParser())
para = para.get_para()

print(para.learning_rate)