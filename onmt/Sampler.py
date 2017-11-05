import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.IO import *
from onmt.Utils import use_gpu
sys.path.insert(0,"./rewards/math/")
from scorer import Math

import codecs

class RewardCriterion(nn.Module):
  def __init__(self):
    super(RewardCriterion, self).__init__()

  def forward(self, input, seq, reward,pad=3):
    mask = (seq!=pad).float()
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).view(-1)
    output = - input * reward 
    output = output * Variable(mask)
    output = torch.sum(output) / torch.sum(mask)

    return output

class MathDataset(onmt.IO.ONMTDataset):

    def __init__(self, src_path, tgt_path, fields, opt,
                 src_img_dir=None, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        src_path: location of source-side data
        tgt_path: location of target-side data or None. If it exists, it
                  source and target data must be the same length.
        fields:
        src_img_dir: if not None, uses images instead of text for the
                     source. TODO: finish
        """
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        if self.type_ == "text":
            self.src_vocabs = []
            src_truncate = 0 if opt is None else opt.src_seq_length_trunc
            src_point = next(self._read_corpus_file(src_path, src_truncate))
            self.nfeatures = src_point[2]
            src_data = self._read_corpus_file(src_path, src_truncate)
            src_examples = self._construct_examples(src_data, "src")
        else:
            # TODO finish this.
            if not transforms:
                load_image_libs()

        if tgt_path is not None:
            tgt_truncate = 0 if opt is None else opt.tgt_seq_length_trunc
            tgt_data = self._read_corpus_file(tgt_path, tgt_truncate,tgt=True)
            # assert len(src_data) == len(tgt_data), \
            #     "Len src and tgt do not match"
            tgt_examples = self._construct_examples(tgt_data, "tgt")
            print(list(tgt_data))
            print(tgt_examples['tgt'])
        else:
            tgt_examples = None

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples is not None:
            examples = (join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        def dynamic_dict(examples):
            for example in examples:
                src = example["src"]
                src_vocab = torchtext.vocab.Vocab(Counter(src))
                self.src_vocabs.append(src_vocab)
                # mapping source tokens to indices in the dynamic dict
                src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
                example["src_map"] = src_map

                if "tgt" in example:
                    tgt = example["tgt"]
                    mask = torch.LongTensor(
                            [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                    example["alignment"] = mask
                yield example

        if opt is None or opt.dynamic_dict:
            examples = dynamic_dict(examples)

        # Peek at the first to see which fields are used.
        ex = next(examples)
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(
            construct_final(chain([ex], examples)),
            fields,
            filter_pred if opt is not None
            else None)

    def _read_corpus_file(self, path, truncate,tgt=False):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)

        returns: (word, features, nfeat) triples for each line
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            if tgt:
              lines = (line.strip() for line in corpus_file)
            else:
              lines = (line.split() for line in corpus_file)
            for line in lines:
                yield line,None,None

  
class MathReward:
  def __init__(self):
    self.m = Math()

  def calc(self,base,batch,eqs):
    brewards = self.m.score(base,eqs)
    grewards = self.m.score(batch,eqs)
    rewards = [grewards[i] - brewards[i] for i in range(len(brewards))]
    m = max([len(x.strip().split(" ")) for x in batch])+1
    rewards = torch.cuda.FloatTensor(rewards).repeat(m,1)
    return rewards, sum(brewards)

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
    self.pad = self.fields['tgt'].vocab.stoi['<blank>']

  #def baseline(self,batch):
  def sample(self,batch,baseline=False):
    src, src_lengths = batch
    batch_size = src.size(1)
    #src2 = onmt.IO.make_features(batch, 'src')
    src = src.unsqueeze(2)
    encStates, context = self.model.encoder(src, src_lengths)
    decStates = self.model.decoder.init_decoder_state(src, context, encStates)
    inp = [[[self.fields['tgt'].vocab.stoi['<s>']]]*batch_size]
    i = 0
    inp = Variable(torch.cuda.LongTensor(inp))
    lprobs = []
    outputs = []
    lengths = []
    while i<self.opt.max_sent_length:
      i+=1
      decOut, decStates, attn = self.model.decoder(inp, context, decStates)
      for x in decOut:
        logprobs = self.model.generator.forward(x)
        if baseline:
          sampleLogprobs, sample = torch.max(logprobs,1)
          sample = sample.data.unsqueeze(1)
        else:
          probs = torch.exp(logprobs).data.cpu()
          #print(torch.min(probs))
          sample = torch.multinomial(probs,1).cuda()
          sampleLogprobs = logprobs.gather(1, Variable(sample, requires_grad=False)) 

      #stop when done
      if i == 1:
        unfinished = sample != self.eos
      else:
        unfinished = unfinished * justfinished
      if unfinished.sum() == 0:
        break
      justfinished = (sample != self.eos)
      sample = sample*unfinished.type_as(sample)
      sample = sample + ((unfinished==0)*self.pad).type_as(sample)
      outputs.append(sample)
      lprobs.append(sampleLogprobs.view(-1))
      inp = Variable(sample.unsqueeze(0))

    return torch.cat(outputs, 1), torch.cat([x.unsqueeze(1) for x in lprobs], 1)

  def decode(self,batch):
    seqlen = batch.size(1)
    sents = []
    for s in batch:
      i = 0
      sent = " "
      while i<seqlen and s[i]!=self.eos:
        sent = sent + " " + self.fields['tgt'].vocab.itos[s[i]]
        i+=1
      sents.append(sent)
    return sents
