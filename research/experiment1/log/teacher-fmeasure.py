#!/usr/local/bin/python3

import json

with open("2.txt") as f:
  lines = f.readlines()
  mat = "".join(lines[:-1])
  mat += lines[-1][:-1]
  mat = mat[:-1]
  a = json.loads(mat)

  prec = [0] * 10
  rec = [0] * 10
  for c in range(10):
    prec[c] =  a[c][c] / sum(a[c])
    rec[c] =  a[c][c] / sum([a[x][c] for x in range(10)])
  acc = sum([a[c][c] for c in range(10)]) / sum([sum(a[i]) for i in range(10)])
  total_prec = sum(prec) / 10
  total_rec = sum(rec) / 10
  fmeasure = 2 * total_prec * total_rec / (total_prec + total_rec)
#  print(acc)
#  print(total_prec)
#  print(total_rec)
  print(fmeasure)
