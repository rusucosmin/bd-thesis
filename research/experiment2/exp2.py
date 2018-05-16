import time
import argparse
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cifar10vgg import cifar10vgg
from student1 import student1

parser = argparse.ArgumentParser(
    description="experiment1 of the research project")
parser.add_argument("-t", "--trainTeacher",
    help="train teacher", action="store_true")
parser.add_argument("-s", "--trainStudents",
    help="train students", action="store_true")
parser.add_argument("-d", "--distillate",
    help="distillate", action="store_true")
#parser.add_argument("-p", "--plot",
#    help="plots the data", action="store_true")

args = parser.parse_args()

if args.trainTeacher:
  t = cifar10vgg(False)
  students = [Student4("student4"), Student5("student5"), Student("student"), Student2("student2"), Student3("student3")]

if args.trainTeacher:
  data = c.trainTeacher()
  if args.plot:
    plotTeacher(data[2])
if args.trainStudents:
  data = c.trainStudents()
  if args.plot:
    plotStudents(data)
if args.distillate:
  data = c.distillate()
  if args.plot:
    plotDistillation(data)

#plotTeacher([1, 2, 3])
#plotStudents([{'name': 'student1', 'data': [[1, 2, 3], [1, 2, 3], [3, 3, 3]]}, {'name': 'student2', 'data': [[], [], [2, 3, 4]]}])
#plotDistillation([{'name': 'student1', 'data': [{'t': 1, 'data': [[], [], [1, 2, 3] ]}, {'t': 2, 'data': [[], [], [2, 3, 4]] } ]},
#                {'name': 'student2', 'data': [{'t': 1, 'data': [[], [], [1, 2, 3] ]}]}])
