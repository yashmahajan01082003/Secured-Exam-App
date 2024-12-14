from django.db import models

# Create your models here.


#College model
class College(models.Model):
    college_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    class Meta:
        verbose_name_plural = "Colleges"


class CollegeSubjects(models.Model):
    subject_code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    branch = models.CharField(max_length=100)
    year = models.IntegerField()
    marks = models.IntegerField()
    time = models.IntegerField()

    class Meta:
        verbose_name_plural = "College Subjects"

    def __str__(self):
        return self.name

class Student(models.Model):
    student_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    branch = models.CharField(max_length=100)
    college = models.ForeignKey(College, on_delete=models.CASCADE)

    class Meta:
        verbose_name_plural = "Student"

    def __str__(self):
        return self.name

class Credentials(models.Model):
    student_id = models.CharField(max_length=10, unique=True)
    password = models.CharField(max_length=15)

    class Meta:
        verbose_name_plural = "Credentials"

    def __str__(self):
        return self.student_id

class Exam(models.Model):
    subject = models.ForeignKey(CollegeSubjects, on_delete=models.CASCADE)
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()

    def __str__(self):
        return f"{self.subject.name} - {self.date}"

    def duration(self):
        return self.end_time - self.start_time

class SubjectSpecificExam(models.Model):
    student = models.ForeignKey(Student,on_delete=models.CASCADE)
    subject = models.ForeignKey(CollegeSubjects,on_delete=models.CASCADE)

    def __str__(self):
        return self.student.student_id


class ContactDetails(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20)

    def __str__(self):
        return self.student.student_id


class ExamRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    warningcount = models.PositiveIntegerField(null=True)
    image = models.ImageField(upload_to='exam_images')
    timeofjoining = models.DateTimeField()
    subject = models.ForeignKey(CollegeSubjects, on_delete=models.CASCADE)
    college = models.ForeignKey(College, on_delete=models.CASCADE)
    is_session_running = models.BooleanField(default=False)
    score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Exam Record: Student {self.student} - Subject {self.subject}"

class ResponsesRec(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    college = models.ForeignKey(College, on_delete=models.CASCADE)
    subject = models.ForeignKey(CollegeSubjects, on_delete=models.CASCADE)
    question_number = models.PositiveIntegerField()
    question = models.TextField()
    response = models.TextField()
    total_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Response: {self.student}-{ self.subject} - Question {self.question_number}"

class StudentPerf(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    college = models.ForeignKey(College, on_delete=models.CASCADE)
    subject = models.ForeignKey(CollegeSubjects, on_delete=models.CASCADE)
    total_score = models.PositiveIntegerField()
    warningcount = models.PositiveIntegerField()

    def __str__(self):
        return f"Exam Record: Student {self.student} - Subject {self.subject}"



