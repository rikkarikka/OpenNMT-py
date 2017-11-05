import sys
import torch
from torch.autograd import Variable
import rik104model
from vecHandler import Vecs

class Math:
  def __init__(self):
    self.model = torch.load("rewards/math/badModel.pt")
    self.model.lstm.flatten_parameters()
    self.model.eval()
    self.vecs = Vecs()
    with open("rewards/math/labels",'r') as f:
      self.labels = [x.strip() for x in f.readlines()]

  def get_vec(self, texts):
    tl = max([len(x) for x in texts])
    inp3d = torch.cuda.FloatTensor(len(texts),tl,300)
    for i in range(len(texts)):
      for j in range(len(texts[i])):
        inp3d[i,j,:] = self.vecs[texts[i][j]] 
    return inp3d

  def score(self,texts,eq):
    vs = self.get_vec(texts)
    l = [self.labels.index(e) for e in eq]
    bsz = 50
    b = 0
    scores = []
    while b < vs.size(0):
      out = self.model(Variable(vs[b:b+bsz], requires_grad=False))
      sm = torch.nn.functional.softmax(out)
      for i,label in enumerate(l[b:b+bsz]):
        scores.append(sm[i,label].data[0])
      b += bsz
    return scores

if __name__=="__main__":
  m = Math()
  x = m.score([["this","is","test","."]],["VAR_0 = NUMBER_1 - NUMBER_2"])
  print(x)


