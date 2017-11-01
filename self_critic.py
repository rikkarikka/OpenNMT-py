import sys
import os
import argparse
import torch
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.Utils import use_gpu
import opts

parser = argparse.ArgumentParser(description='self_critic.py')
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()

if __name__=="__main__":
  s = onmt.Sampler(opt)
