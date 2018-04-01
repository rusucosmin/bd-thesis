from teacher import Teacher
from student import Student
from student2 import Student2
from student3 import Student3
from controller import Controller

t = Teacher("teacher")
students = [Student3("student3"), Student("student"), Student2("student2")]
c = Controller(t, students)

c.trainTeacher()
c.trainStudents()

