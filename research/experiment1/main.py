from teacher import Teacher
from student import Student
from student2 import Student2
from controller import Controller

t = Teacher("teacher")
students = [Student2("student2"), Student("student")]
c = Controller(t, students)

#c.trainTeacher()
c.trainStudents()

