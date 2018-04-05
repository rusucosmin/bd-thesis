from teacher import Teacher
from student import Student
from student2 import Student2
from student3 import Student3
from controller import Controller

t = Teacher("teacher")
students = [Student("student")] #, Student2("student2"), Student3("student3")]
c = Controller(t, students)

c.trainTeacher()
c.trainStudents()
c.distillate()

