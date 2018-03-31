from teacher import Teacher
from student import Student
from controller import Controller

t = Teacher("teacher")
students = [Student("student")]
c = Controller(t, students)

c.trainStudents()

