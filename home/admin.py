from django.contrib import admin

# Register your models here.

from home.models import College, Exam, Credentials, Student, CollegeSubjects, SubjectSpecificExam, ContactDetails, \
    ExamRecord, ResponsesRec, StudentPerf

# Register your models here.
admin.site.register(College)
admin.site.register(CollegeSubjects)
admin.site.register(Student)
admin.site.register(Credentials)
admin.site.register(Exam)
admin.site.register(SubjectSpecificExam)
admin.site.register(ContactDetails)
admin.site.register(ExamRecord)
admin.site.register(ResponsesRec)
admin.site.register(StudentPerf)