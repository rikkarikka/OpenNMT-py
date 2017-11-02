import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.Utils import use_gpu

class RewardCriterion(nn.Module):
  def __init__(self):
    super(RewardCriterion, self).__init__()

  def forward(self, input, seq, reward):
    mask = (seq>0).float()
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).view(-1)
    output = - input * reward 
    output = output * Variable(mask)
    output = torch.sum(output) / torch.sum(mask)

    return output

class Rewarder:
  def __init__(self):
    pass

  def calc(self,batch):
    rewards = batch == 5
    return rewards.float()

class Sampler(object):
  def __init__(self, opt, dummy_opt={}):
    # Add in default model arguments, possibly added since training.
    self.opt = opt
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    self.fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    self._type = model_opt.encoder_type
    self.copy_attn = model_opt.copy_attn

    self.model = onmt.ModelConstructor.make_base_model(
                        model_opt, self.fields, use_gpu(opt), checkpoint)

    self.eos = self.fields['tgt'].vocab.stoi['</s>']

  def sample(self,batch):
    batch_size = batch.batch_size

    # (1) Run the encoder on the src.
    _, src_lengths = batch.src
    src = onmt.IO.make_features(batch, 'src')
    encStates, context = self.model.encoder(src, src_lengths)
    decStates = self.model.decoder.init_decoder_state(src, context, encStates)
    inp = [[[self.fields['tgt'].vocab.stoi['<s>']]]*batch_size]
    i = 0
    inp = Variable(torch.cuda.LongTensor(inp))
    lprobs = []
    outputs = []
    done = []
    while i<self.opt.max_sent_length:
      i+=1
      decOut, decStates, attn = self.model.decoder(inp, context, decStates)
      for x in decOut:
        logprobs = self.model.generator.forward(x)
        probs = torch.exp(logprobs).data.cpu()
        sample = torch.multinomial(probs,1).cuda()
        sampleLogprobs = logprobs.gather(1, Variable(sample, requires_grad=False)) 

      #stop when done
      if i == 1:
        unfinished = sample != self.eos
      else:
        unfinished = unfinished * (sample != self.eos)
      if unfinished.sum() == 0:
        break
      outputs.append(sample*unfinished.type_as(sample)) 
      lprobs.append(sampleLogprobs.view(-1))
      inp = Variable(sample.unsqueeze(0))

    return torch.cat(outputs, 1), torch.cat([x.unsqueeze(1) for x in lprobs], 1)

