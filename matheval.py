import sys
with open("draw_data/draw-dev.src") as f:
  tgts = [' '.join(x.split()) for x in f.read().strip().split('\n')]

with open(sys.argv[1]) as f:
  preds = f.read().strip().split('\n')

print(sum([preds[i]==tgts[i] for i in range(len(preds))]))
