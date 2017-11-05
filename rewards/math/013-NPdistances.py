import sys
import pickle
import linecache
import numpy as np

class Embeddings:
  def __init__(self,d):
    self.lines = self.mklines(d)
    self.d = d
    self.cache = {}
    self.vecs = None
    self.words = None
    self.norms = None

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
      
      
try: 
  with open("sw.pkl",'rb') as f:
    e = pickle.load(f)
except:
  glove = "/Users/rikka/Tools/gloveVecs/glove.6B.300d.txt"
  e = Embeddings(glove)

  with open("sw.nounphrases.txt") as f:
    for l in f:
      e.get(l.lower().strip())
  with open("sw.pkl",'wb') as f:
    pickle.dump(e,f)

with open("math.nounphrases.txt") as f:
  maths = [''.join([y for y in x if y not in "0123465789.,"]) for x in f.read().strip().split('\n')]
for m in maths:
  e.closest(m)
  print()
