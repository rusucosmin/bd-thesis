#!/usr/local/bin/python3

import json

models = ["student", "student2", "student3", "student4",
    "student5"]
T = [1, 3, 6, 7, 8, 9, 10, 11, 12, 15, 20]
with open("1.txt") as f:
  lines = f.readlines()
  arr = "[ "
  for (i, model) in enumerate(models):
    for (j, t) in enumerate(T):
      st = (i * 11 + j) * 10
      dr = st + 10
      print(st, dr)
      mat = "".join(lines[st:dr-1])
      mat += lines[dr-1][:-1]
      print(">>>")
      print(mat)
      print("<<<")
      arr += mat
  arr = arr[:-1] + " ]"
  arr = json.loads(arr)

  ind = 0
  for (i, model) in enumerate(models):
    for (j, t) in enumerate(T):
      a = arr[ind]
      rec = [0] * 10
      for c in range(10):
        rec[c] =  a[c][c] / sum([a[x][c] for x in range(10)])
      total_rec = sum(rec) / 10
      print(total_rec)
      ind += 1
