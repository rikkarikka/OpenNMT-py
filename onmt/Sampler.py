import torch
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.Utils import use_gpu


class Sampler(onmt.Translator):
  def __init__(self, opt, dummy_opt={}):
    super(Sampler,self).__init__(opt,dummy_opt)

  def sample(self,batch):
    beam_size = self.opt.beam_size
    batch_size = batch.batch_size

    # (1) Run the encoder on the src.
    _, src_lengths = batch.src
    src = onmt.IO.make_features(batch, 'src')
    encStates, context = self.model.encoder(src, src_lengths)
    decStates = self.model.decoder.init_decoder_state(src, context, encStates)

    
