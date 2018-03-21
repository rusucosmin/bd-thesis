import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Controller:
  def __init__(self, teacher, students):
    print("Controller::__init__")
    self.teacher = teacher
    self.students = students

    print("> Loading MNIST data...")
    self.mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

  def trainTeacher(self):
    print("trainingTeacher")

  def distillate(self, teacher, student):
    print("distillating %s to %s" % (teacher, student))

  def run(self):
    print("Controller::run")
    self.teacher.train(self.mnist)
