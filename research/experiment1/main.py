from teacher import Teacher
from controller import Controller

t = Teacher("teacher")
students = []
c = Controller(t, students)

c.run()

