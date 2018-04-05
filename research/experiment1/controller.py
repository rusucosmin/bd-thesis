import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Controller:
  def __init__(self, teacher, students):
    print("Controller::__init__")
    self.teacher = teacher
    self.students = students

    print("> Loading MNIST data...")
    self.mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

  def trainStudents(self):
    print("trainingStudents")
    for student in self.students:
      student.train(self.mnist)

  def trainTeacher(self):
    print("trainingTeacher")
    self.teacher.train(self.mnist)

  def distillate(self):
    print("distillating")
    T = [1, 3, 6, 9, 10]
    self.teacher.softTargets(T, self.mnist)
    for student in self.students:
      for t in T:
        student.distillate(
            self.mnist,
            np.load("soft-targets-%d" % t),
            t)

  def run(self):
    print("Controller::run")
