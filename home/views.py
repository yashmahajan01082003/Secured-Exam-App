import csv
import math
import time
from ctypes import cast, POINTER
from django.shortcuts import render, redirect, get_object_or_404
import cvzone
import mediapipe as mp
import pythoncom
import wmi
from comtypes import CLSCTX_ALL
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime, date
import os
import cv2
import threading
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import os
import csv
import cv2
import numpy as np
import pyautogui
from .models import Student, CollegeSubjects, ExamRecord, Exam, ResponsesRec

import pyautogui
import numpy as np
import threading

from pycaw.api.endpointvolume import IAudioEndpointVolume
from pycaw.utils import AudioUtilities

from core import settings
from home.models import Credentials, SubjectSpecificExam, Student, CollegeSubjects, College, Exam, ContactDetails, \
    ExamRecord, ResponsesRec, StudentPerf


# Create your views here.

def index(request):
    College1 = College.objects.all()
    collen= 0
    for Col in College1:
        collen += 1

    CollegeSubjects1 = CollegeSubjects.objects.all()
    colsublen = 0
    for ColSub in CollegeSubjects1:
        colsublen += 1

    studlen = Student.objects.count()

    studperf = StudentPerf.objects.count()


    context={
        "collen" : collen,
        "colsublen" : colsublen,
        "studlen" : studlen,
        "College" : College1,
        "ColSub" : CollegeSubjects1,
        "studperf" : studperf,
    }

    # Page from the theme 
    return render(request, 'pages/index.html',context)


#Student login page appears
def loginasuser(request):
    return render(request, 'signin/loginuser.html')

#Student login actually takes place
def loginasstud(request):
    studentid = request.POST.get('studentid')
    password = request.POST.get('password')
    User = Credentials.objects.filter(student_id=studentid,password=password).first()

    if User:
        name = Student.objects.filter(student_id=studentid).first()
        College1 = College.objects.filter(name=name.college).first()
        ContactDets = ContactDetails.objects.filter(student=name).first()
        Subjects = SubjectSpecificExam.objects.filter(student = name)




        cdate = date.today()
        ctime = datetime.now().time()

        buff = []
        allSubs = []
        for sub in Subjects:
            exam = Exam.objects.filter(subject=sub.subject).first()
            examdate = exam.date
            if examdate ==  cdate:
                if exam.start_time > ctime :
                    buffer = 1
                if exam.start_time <= ctime and  exam.end_time > ctime:
                    buffer = 0
                if exam.start_time <= ctime and  exam.end_time < ctime:
                    buffer = -1
            else :
                buffer = 1

            buff.append(buffer)
            allSubs.append(exam)
        Pair = zip(allSubs, buff)

        context  = {
            'Subjects' : allSubs,
            'Student' : name,
            'College' : College1,
            'Contact' : ContactDets,
            'Paired' : Pair,
        }

        return render(request, 'displayprofile/profile.html',context)
    else:
        return render(request, 'signin/loginuser.html')


class MCQ:
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None

def execQuiz(request, qName, student_id, qNo, flag):
    def capture_and_insert_image(student_id, qName):
        while True:
            # Define the output image dimensions
            output_width = 800
            output_height = 600

            # Calculate the aspect ratios
            screen_aspect_ratio = pyautogui.size()[0] / pyautogui.size()[1]
            webcam_aspect_ratio = 4 / 3  # Assuming 4:3 aspect ratio for the webcam frame

            # Calculate the resized dimensions while maintaining aspect ratios
            resized_screen_width = output_width // 2
            resized_screen_height = int(resized_screen_width / screen_aspect_ratio)
            resized_webcam_width = output_width // 2
            resized_webcam_height = int(resized_webcam_width / webcam_aspect_ratio)

            # Capture screenshot
            screenshot = pyautogui.screenshot()
            screen_frame = np.array(screenshot)
            resized_screen_frame = cv2.resize(screen_frame, (resized_screen_width, resized_screen_height))

            # Capture webcam frame
            webcam = cv2.VideoCapture(0)
            ret, webcam_frame = webcam.read()
            resized_webcam_frame = cv2.resize(webcam_frame, (resized_webcam_width, resized_webcam_height))

            # Create combined frame
            combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            screen_y = (output_height - resized_screen_height) // 2
            webcam_y = (output_height - resized_webcam_height) // 2
            combined_frame[screen_y:screen_y + resized_screen_height, :resized_screen_width, :] = resized_screen_frame
            combined_frame[webcam_y:webcam_y + resized_webcam_height, output_width // 2:output_width // 2 + resized_webcam_width, :] = resized_webcam_frame

            # Retrieve Values from database
            student = Student.objects.filter(student_id=student_id).first()
            college = College.objects.filter(name=student.college).first()
            subject = CollegeSubjects.objects.filter(name=qName).first()

            # Save combined frame as an image
            student_folder = os.path.join(settings.MEDIA_ROOT, 'exam_images', str(student.student_id))
            os.makedirs(student_folder, exist_ok=True)
            subjectspecific = os.path.join(student_folder, str(qName))
            os.makedirs(subjectspecific, exist_ok=True)

            image_name = f"{student.name} {student.student_id}{qName} {datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = os.path.join(settings.MEDIA_ROOT, f'exam_images/{student.student_id}/{qName}', image_name)
            cv2.imwrite(image_path, combined_frame)

            # Check the entry of user if his session has started
            ExamRecord1 = ExamRecord.objects.filter(student=student).last()

            # Create an ExamRecord object and set the image field
            if ExamRecord1:
                exam_record = ExamRecord.objects.create(
                    student=student,
                    image=image_path,
                    timeofjoining=ExamRecord1.timeofjoining,
                    subject=ExamRecord1.subject,
                    college=ExamRecord1.college,
                    is_session_running=ExamRecord1.is_session_running,
                    score=ExamRecord1.score,
                )
                exam_record.save()
            else:
                exam_record = ExamRecord.objects.create(
                    student=student,
                    image=image_path,
                    timeofjoining=datetime.now(),
                    subject=subject,
                    college=college,
                    is_session_running=True,
                    score=0,
                )
                exam_record.save()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Clean up resources
        webcam.release()
        cv2.destroyAllWindows()

    # Retrieve Values from database
    student = Student.objects.filter(student_id=student_id).first()
    college = College.objects.filter(name=student.college).first()
    subject = CollegeSubjects.objects.filter(name=qName).first()

    capture_thread = threading.Thread(target=capture_and_insert_image, args=(student_id, qName))

    # Start the thread
    capture_thread.start()

    # Continue with the main program
    print("Main program continues...")

    # Check if final submit button is pressed
    if 'final_submit' in request.POST:
        return redirect('show_result', qName=qName, student_id=student_id)

    # Check the entry of user if his session has started
    ExamRecord1 = ExamRecord.objects.filter(student=student).last()

    # Import csv file data
    pathCSV = "questionset/" + qName + '.csv'
    with open(pathCSV, newline='\n') as f:
        reader = csv.reader(f)
        dataAll = list(reader)[1:]

    # Create object for each mcq
    mcqList = [MCQ(q) for q in dataAll]

    if flag == 1:
        if qNo >= 0 and qNo < len(mcqList) - 1:
            counter = qNo
            counter += 1
            qNo = counter

    if flag == 2:
        if qNo > 1 and qNo < len(mcqList):
            counter = qNo
            counter -= 1
            qNo = counter

    if qNo == len(mcqList) - 1:
        lastsubmit = 1
    else:
        lastsubmit = 0

    RespRec1 = ResponsesRec.objects.filter(subject=subject, student=student, question_number=qNo).first()
    if RespRec1:
        res = RespRec1.response
    else:
        res = 0

    mcqlen = len(mcqList)
    progp = (qNo + 1) / mcqlen * 100

    Exam1 = Exam.objects.filter(subject=subject).first()

    context = {
        'MCQ': mcqList[qNo],
        'Stud': student,
        'qNo': qNo,
        'subject': qName,
        'resp': res,
        'mcql': mcqlen,
        'progesspercent': progp,
        'fsubmit': lastsubmit,
        'endTime': Exam1.end_time,
        'date': Exam1.date,
    }
    print(Exam1.end_time, Exam1.date)

    return render(request, 'quiz page/quiz1.html', context)


def storeRes(request,qName,student_id,qNo,question,resp):
    class MCQ():
            def __init__(self, data):
                self.question = data[0]
                self.choice1 = data[1]
                self.choice2 = data[2]
                self.choice3 = data[3]
                self.choice4 = data[4]
                self.answer = int(data[5])
                self.userAns = None

            # Import csv file data

            # Retrieve Values from database

    student = Student.objects.filter(student_id=student_id).first()
    college = College.objects.filter(name=student.college).first()
    subject = CollegeSubjects.objects.filter(name=qName).first()

    pathCSV = "questionset/" + qName + '.csv'
    with open(pathCSV, newline='\n') as f:
        reader = csv.reader(f)
        dataAll = list(reader)[1:]

    # Create object for each mcq
    mcqList = [0, ]
    for i, q in enumerate(dataAll):
        mcq = MCQ(q)
        mcqList.append(mcq)

    RespRec1 = ResponsesRec.objects.filter(subject=subject,student = student, question_number = qNo).first()

    if RespRec1:
        if mcqList[qNo].answer == resp:
            RespRec1.question = mcqList[qNo].question
            RespRec1.response = resp
            RespRec1.total_score = 1
            RespRec1.save()
        else:
            RespRec1.question = mcqList[qNo].question
            RespRec1.response = resp
            RespRec1.total_score = 0
            RespRec1.save()

    else:
        if mcqList[qNo].answer == resp:
            RespRec1 = ResponsesRec.objects.create(
                student = student,
                college = college,
                subject = subject,
                question_number = qNo,
                question = mcqList[qNo].question,
                total_score = 1,
                response = resp,
            )
            RespRec1.save()

        else:
            RespRec1 = ResponsesRec.objects.create(
                student=student,
                college=college,
                subject=subject,
                question_number=qNo,
                question=mcqList[qNo].question,
                total_score=0,
                response=resp,
            )
            RespRec1.save()

    mcqlen = len(mcqList)
    progp = (qNo+1) / mcqlen*100


    student = Student.objects.filter(student_id=student_id).first()
    college = College.objects.filter(name=student.college).first()
    subject = CollegeSubjects.objects.filter(name=qName).first()

    Exam1 = Exam.objects.filter(subject=subject).first()
    stime = Exam1.start_time
    etime = Exam1.end_time


    if qNo == len(mcqList)-1:
        lastsubmit = 1
    else:
        lastsubmit = 0


    context = {
        'MCQ': mcqList[qNo],
        'Stud': student,
        'qNo': qNo,
        'subject': qName,
        'resp' : resp,
        'mcql' : mcqlen,
        'progesspercent' : progp,
        'stime' : stime ,
        'etime' : etime,
        'fsubmit': lastsubmit,
        'endTime': Exam1.end_time,
        'date': Exam1.date,

    }
    print(Exam1.end_time , Exam1.date)

    return render(request, 'quiz page/quiz1.html', context)


def execWarn(request, qName, student_id , qNo):
    class MCQ():
            def __init__(self, data):
                self.question = data[0]
                self.choice1 = data[1]
                self.choice2 = data[2]
                self.choice3 = data[3]
                self.choice4 = data[4]
                self.answer = int(data[5])
                self.userAns = None

    student = Student.objects.filter(student_id=student_id).first()
    college = College.objects.filter(name=student.college).first()
    subject = CollegeSubjects.objects.filter(name=qName).first()



    ExamRecord1 = ExamRecord.objects.filter(student=student, subject = subject)
    for er in ExamRecord1:
        if er.warningcount:
            er.warningcount = er.warningcount + 1
        else:
            er.warningcount=1
        er.save()

    pathCSV = "questionset/" + qName + '.csv'
    with open(pathCSV, newline='\n') as f:
        reader = csv.reader(f)
        dataAll = list(reader)[1:]

    # Create object for each mcq
    mcqList = [0, ]
    for i, q in enumerate(dataAll):
        mcq = MCQ(q)
        mcqList.append(mcq)


    mcqlen = len(mcqList)
    progp = (qNo + 1) / mcqlen * 100

    lastsubmit = mcqlen-1

    Exam1 = Exam.objects.filter(subject = subject).first()

    context = {
        'MCQ': mcqList[qNo],
        'Stud': student,
        'qNo': qNo,
        'subject': qName,
        'mcql': mcqlen,
        'progesspercent': progp,
        'fsubmit': lastsubmit,
        'endTime' : Exam1.end_time,
        'date' : Exam1.date,

    }
    print(Exam1.end_time , Exam1.date)
    return render(request, 'quiz page/quiz1.html', context)

def execFinal(request,qName,student_id):
    student = Student.objects.filter(student_id=student_id).first()
    college = College.objects.filter(name=student.college).first()
    subject = CollegeSubjects.objects.filter(name=qName).first()
    resprec = ResponsesRec.objects.filter(student=student,subject=subject)
    ExamRecord1 = ExamRecord.objects.filter(student=student, subject = subject).first()
    i = 0
    for var in resprec:
        i = i + var.total_score
    totalScore = i

    totalWarnings = ExamRecord1.warningcount

    StudentPerf1 = StudentPerf.objects.create(
        student = student,
        college = college,
        subject = subject,
        total_score = totalScore,
        warningcount = totalWarnings
    )
    studentid = request.POST.get('studentid')
    password = request.POST.get('password')
    User = Credentials.objects.filter(student_id=student_id, password=password).first()


    cdate = date.today()
    ctime = datetime.now().time()

    name = Student.objects.filter(student_id=student_id).first()
    College1 = College.objects.filter(name=name.college).first()
    ContactDets = ContactDetails.objects.filter(student=name).first()
    Subjects = SubjectSpecificExam.objects.filter(student=name)

    buff = []
    allSubs = []
    for sub in Subjects:
        exam = Exam.objects.filter(subject=sub.subject).first()
        examdate = exam.date
        if examdate == cdate:
            if exam.start_time > ctime:
                 buffer = 1
            if exam.start_time <= ctime and exam.end_time > ctime:
                buffer = 0
            if exam.start_time <= ctime and exam.end_time < ctime:
                    buffer = -1
        else:
            buffer = 1

            buff.append(buffer)
            allSubs.append(exam)


    Pair = zip(allSubs, buff)

    context = {
        'Subjects': allSubs,
        'Student': name,
        'College': College1,
        'Contact': ContactDets,
        'Paired': Pair,
    }


    return render(request, 'displayprofile/profile.html', context)

@login_required
def fetchdata(request):
    return render(request,'fetchdata/index.html')

@login_required
def getRoll(request):
    roll = request.GET.get('roll')
    StudentObj = Student.objects.filter(student_id=roll).first()
    SubjectSpecificExamObj = SubjectSpecificExam.objects.filter(student=StudentObj)
    # Get all unique subjects for the student
    AttemptedSubject = ExamRecord.objects.filter(student=StudentObj).distinct()
    list =[]
    for subs in AttemptedSubject:
        if subs.subject in list:
            pass
        else:
            list.append(subs.subject)
    AttemptedSubjects = list
    context={
        'Student' : StudentObj,
        'AllSubjects' : SubjectSpecificExamObj,
        'ASubs' : AttemptedSubjects,
    }
    return render(request,'fetchdata/displaystud.html' , context)


def examScore(request,roll,sub):
    StudentObj = Student.objects.filter(student_id=roll).first()
    SubjectSpecificExamObj = SubjectSpecificExam.objects.filter(student=StudentObj)
    # Get all unique subjects for the student
    Subject1 = CollegeSubjects.objects.filter(name=sub).first()
    AttemptedSubject = ExamRecord.objects.filter(student=StudentObj).distinct()
    StudentPerf1 = StudentPerf.objects.filter(student=StudentObj,subject=Subject1)
    ResponsesRec1 = ResponsesRec.objects.filter(student=StudentObj,subject=Subject1)
    len1 = len(ResponsesRec1)
    context={
        'ASubs' : StudentPerf1,
        'Student': StudentObj,
        'sub': sub,
        'QandA' : ResponsesRec1,
        'len1':len1,
    }
    return render(request,'fetchdata/displayscore.html',context)

@login_required
def examRecs(request,roll,sub):
    StudentObj = Student.objects.filter(student_id=roll).first()
    SubjectSpecificExamObj = SubjectSpecificExam.objects.filter(student=StudentObj)
    # Get all unique subjects for the student
    AttemptedSubjects = ExamRecord.objects.filter(student=StudentObj).distinct()
    context = {
        'Student': StudentObj,
        'AllSubjects': SubjectSpecificExamObj,
        'ASubs': AttemptedSubjects,
        'sub' :sub,
    }
    return render(request,'fetchdata/displayexamrec.html' , context)

def execVol(request):
    pythoncom.CoInitialize()
    device = AudioUtilities.GetSpeakers()
    interface = device.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    volume.SetMasterVolumeLevel(0, None)
    minVol = volRange[0]
    maxVol = volRange[1]
    print(minVol, maxVol)
    vol = 0
    volBar = 400
    volPer = 0

    # dimensions
    wCm, hCam = 1200, 720

    cap = cv2.VideoCapture(0)
    cap.set(3, wCm)
    cap.set(4, hCam)
    pTime = 0

    detector = handDetector(detectionCon=0.7)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[4],lmList[8])

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand range 10 - 200
            # Volume Range -65  - 0
            vol = np.interp(length, [50, 250], [minVol, maxVol])
            volBar = np.interp(length, [50, 250], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            print(vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f' FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        key = cv2.waitKey(1)
        if key is ord('q'):
            break
    cv2.destroyAllWindows()  # Close OpenCV windows
    return render(request, 'signin/loginuser.html')


def execBr(request):
    # Define the maximum and minimum brightness values
    MAX_BRIGHTNESS = 100
    MIN_BRIGHTNESS = 0
    pythoncom.CoInitialize()
    # Connect to WMI
    wmi_obj = wmi.WMI(namespace="wmi")

    # Function to set the screen brightness
    def set_brightness(value):
        # Clamp the value within the brightness range
        brightness = max(min(value, MAX_BRIGHTNESS), MIN_BRIGHTNESS)

        # Scale the brightness value to the range [0, 100]
        brightness = int(brightness * 100 / MAX_BRIGHTNESS)

        # Set the screen brightness
        wmi_obj.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(brightness, 0)

    cap = cv2.VideoCapture(0)
    hd = HandDetector()
    val = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        hands, img = hd.findHands(img)

        if hands:
            lm = hands[0]['lmList']

            length, info, img = hd.findDistance(lm[8][0:2], lm[4][0:2], img)
            blevel = np.interp(length, [25, 145], [0, 100])
            val = np.interp(length, [0, 100], [400, 150])
            blevel = int(blevel)

            set_brightness(blevel)

            cv2.rectangle(img, (20, 150), (85, 400), (0, 255, 255), 4)
            cv2.rectangle(img, (20, int(val)), (85, 400), (0, 0, 255), -1)
            cv2.putText(img, str(blevel) + '%', (20, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'signin/loginuser.html')

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon= detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                #mediapipe method that helps to draw points of hands
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNo=0,draw=True):
            lmList=[]
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #Display FPS
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key is ord('q'):
           break


if __name__=="__main__":
    main()

def execFaceMesh(request):
    class FaceMeshDetector():
        def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5):

            self.static_image_mode = False,
            self.max_num_faces = 1,
            self.refine_landmarks = False,
            self.min_detection_confidence = 0.5,
            self.min_tracking_confidence = 0.5

            self.mpDraw = mp.solutions.drawing_utils
            self.mpFaceMesh = mp.solutions.face_mesh
            self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
            self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        def findFaceMesh(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                               self.drawSpec)

                    for id, lm in enumerate(faceLms.landmark):
                        print(lm)
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        print(id, x, y)
            return img

    def main():
        cap = cv2.VideoCapture(0)
        pTime = 0
        detector = FaceMeshDetector()
        while True:
            success, img = cap.read()
            img = detector.findFaceMesh(img)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    main()
    return render(request, 'signin/loginuser.html')

def execFaceDistance(request):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            # cv2.line(img,pointLeft,pointRight,(0,200,0),3)
            # cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)
            # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding the focal length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding the distance or depth
            f = 500
            d = (W * f) / w
            print(d)

            cvzone.putTextRect(img, f'Depth : {int(d)} cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return render(request, 'signin/loginuser.html')

