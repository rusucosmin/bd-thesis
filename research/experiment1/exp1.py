import argparse

from teacher import Teacher
from student import Student
from student2 import Student2
from student3 import Student3
from controller import Controller

parser = argparse.ArgumentParser(
    description="experiment1 of the research project")
parser.add_argument("-t", "--trainTeacher",
    help="train teacher", action="store_true")
parser.add_argument("-s", "--trainStudents",
    help="train students", action="store_true")
parser.add_argument("-d", "--distillate",
    help="distillate", action="store_true")

args = parser.parse_args()

t = Teacher("teacher")
students = [Student("student"), Student2("student2"), Student3("student3")]
c = Controller(t, students)
if args.trainTeacher:
  c.trainTeacher()
elif args.trainStudents:
  c.trainStudents()
elif args.distillate:
  c.distillate()

