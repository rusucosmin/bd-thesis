import time
import argparse
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from teacher import Teacher
from student import Student
from student2 import Student2
from student3 import Student3
from controller import Controller

def plotTeacher(test_accs):
  plt.style.use('ggplot')
  plt.title("Accuracy of teacher")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.plot(range(1, len(test_accs)+1), test_accs, label='Accuracy')
  plt.legend()
  plt.savefig("%d_teacher_accuracy.png" % int(time.time()))
  plt.plot()

def plotStudents(data):
  plt.style.use('ggplot')
  plt.clf()
  plt.title("Accuracy of students")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  for student_data in data:
    name = student_data['name']
    train_data= student_data['data']
    plt.plot(range(1, len(train_data[2])+1), train_data[2], label='Accuracy of %s' % name)
  plt.legend()
  plt.savefig("%d_students_accuracy.png" % int(time.time()))

def plotDistillation(data):
  for student_data in data:
    name = student_data['name']
    plt.style.use('ggplot')
    plt.clf()
    plt.title("Accuracy of %s" % name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for exp in student_data['data']:
      t = exp['t']
      train_data = exp['data']
      plt.plot(range(1, len(train_data[2])+1), train_data[2], label='Accuracy of %s at t = %d' % (name, t))
    plt.legend()
    plt.savefig("%d_distillation_%s_accuracy.png" % (int(time.time()), name))

parser = argparse.ArgumentParser(
    description="experiment1 of the research project")
parser.add_argument("-t", "--trainTeacher",
    help="train teacher", action="store_true")
parser.add_argument("-s", "--trainStudents",
    help="train students", action="store_true")
parser.add_argument("-d", "--distillate",
    help="distillate", action="store_true")
parser.add_argument("-p", "--plot",
    help="plots the data", action="store_true")

args = parser.parse_args()

if args.trainTeacher or args.trainStudents or args.distillate:
  t = Teacher("teacher")
  students = [Student("student"), Student2("student2"), Student3("student3")]
  students = [Student2("student2"), Student3("student3")]
  c = Controller(t, students, verbose=True)
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
