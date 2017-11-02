import sys
import os
import argparse
import torch
from torch import optim
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
  s = onmt.Sampler.Sampler(opt)
  optimizer = optimizer = optim.Adam(s.model.parameters(), lr=0.0001)
  data = onmt.IO.ONMTDataset(opt.src, opt.tgt, s.fields, None)
  test_data = onmt.IO.OrderedIterator(
      dataset=data, device=opt.gpu,
      batch_size=opt.batch_size, train=True , sort=False,
      shuffle=False)
  rewarder = onmt.Sampler.Rewarder()
  rl_crit = onmt.Sampler.RewardCriterion()
  for batch in test_data:
    for k in range(100):
      outputs, logprobs = s.sample(batch)
      reward = rewarder.calc(outputs)
      loss = rl_crit(logprobs, outputs, Variable(reward, requires_grad=False))
      print(reward.sum())
      loss.backward()
      #utils.clip_gradient(optimizer, opt.grad_clip)
      optimizer.step()
    exit()
