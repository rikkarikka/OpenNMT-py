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
  data = onmt.Sampler.MathDataset(opt.src, None, s.fields, None)
  test_data = onmt.IO.OrderedIterator(
      dataset=data, device=opt.gpu,
      batch_size=opt.batch_size, train=True , sort=False,
      shuffle=False)
  rewarder = onmt.Sampler.MathReward()
  rl_crit = onmt.Sampler.RewardCriterion()
  with open(opt.tgt) as f:
    tgts = f.read().strip().split('\n')
  tctr = 0
  for batch in test_data:
    targets = tgts[tctr:tctr+opt.batch_size]
    tctr+=opt.batch_size
    for k in range(1000):
      optimizer.zero_grad()
      outputs, logprobs = s.sample(batch.src)
      base, probs = s.sample((Variable(batch.src[0].data,volatile=True),batch.src[1]),baseline=True)
      bsentences = s.decode(base)
      #print(bsentences)
      gsentences = s.decode(outputs)
      reward,basescore = rewarder.calc(bsentences,gsentences,targets)
      loss = rl_crit(logprobs, outputs, Variable(reward, requires_grad=False),s.pad)
      print(basescore)
      loss.backward()
      #utils.clip_gradient(optimizer, opt.grad_clip)
      optimizer.step()
    exit()
