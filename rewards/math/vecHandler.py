import torchtext
import torch
import sys
import pickle
import linecache
import numpy as np

class Vecs:
  def __init__(self):
    d = "/home/rikka/Tools/gloveVecs/glove.6B.300d.txt"
    self.lines = self.mklines(d)
    self.d = d
    self.cache = {}
    self.vecs = None
    self.words = None
    self.norms = None
  
  def __getitem__(self, w):
    try:
      x = self.cache[w]
    except:
      if w not in self.lines:
        x = np.random.uniform(-1,1,300)
      else:
        x = linecache.getline(self.d,self.lines[w])
        x = x.strip()
        x = [float(y) for y in x.split()[1:]]
      x = torch.cuda.FloatTensor(x)
      self.cache[w] = x
    return x

  def closest(self,phrase):
    print(" PHRASE : ",phrase)
    if self.vecs is None:
      self.words, vecs = zip(*self.cache.items())
      self.vecs = np.array(vecs)
      self.norms = np.linalg.norm(self.vecs,axis=1)
    phrasev = self.get(phrase)
    if phrasev is None: 
      print('no embedding')
      return
    phrasen = np.linalg.norm(phrasev)
    phrasenorms = self.norms*phrasen
    sim = np.divide(self.vecs.dot(phrasev),phrasenorms)
    for i in sim.argsort()[-10:]:
      print(sim[i],self.words[i])
    

  def get(self,phrase):
    phrase = phrase.strip()
    if phrase in self.cache:
      return self.cache[phrase]
    val = []
    for w in phrase.split(" "):
      if w not in self.lines:
        return None
      else:
        x = linecache.getline(self.d,self.lines[w])
        x = x.strip()
        x = [float(y) for y in x.split()[1:]]
      val.append(x)
    val = np.array(val)
    m = np.squeeze(val.mean(0))
    self.cache[phrase] = m
    return m
        
  def mklines(self,d):
    try:
      with open(d+".vocab") as f:
        lines = {k:i+1 for i,k in enumerate(f.read().split("\n"))}
    except:
      with open(d) as f:
        lines = {k.split()[0]:i for i,k in enumerate(f.read().strip().split("\n"))}
      with open(d+".vocab",'w') as f:
        f.write("\n".join(lines))
    return lines

if __name__=="__main__":
  v = Vecs()
  print(v['house'])
  print(v['<person>'])
  print(v['<person>'])
