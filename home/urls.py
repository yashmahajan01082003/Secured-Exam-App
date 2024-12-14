from django.urls import path

from . import views
from core import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', views.index, name='index'),
    path('loginasuser',views.loginasuser, name='loginasuser'),
    path('loginasstud', views.loginasstud, name='loginasstud'),
    path('execQuiz/<str:qName>/<str:student_id>/<int:qNo>/<int:flag>/', views.execQuiz, name='execQuiz'),
    path('storeRes/<str:qName>/<str:student_id>/<int:qNo>/<str:question>/<int:resp>/', views.storeRes, name='storeRes'),
    path('execWarn/<str:qName>/<str:student_id>/<int:qNo>/' , views.execWarn,name='execWarn'),
    path('fetchdata', views.fetchdata, name='fetchdata'),
    path('getRoll', views.getRoll, name='getRoll'),
    path('examRecs/<str:roll>/<str:sub>/', views.examRecs, name='examRecs'),
    path('examScore/<str:roll>/<str:sub>/', views.examScore, name='examScore'),
    path('in', views.index, name='in'),
    path('execVol', views.execVol, name='execVol'),
    path('execBr', views.execBr, name='execBr'),
    path('execFaceMesh', views.execFaceMesh, name='execFaceMesh'),
    path('execFaceDistance', views.execFaceDistance, name='execFaceDistance'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)