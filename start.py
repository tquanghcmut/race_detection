import subprocess

# Định nghĩa đường dẫn đến các script
face_recognition_script = '/Users/twang/PycharmProjects/race_detection_official/run/face_recognition.py'
prediction_script = '/Users/twang/PycharmProjects/race_detection_official/run/prediction.py'

# Chạy face recognition script
subprocess.run(["python", face_recognition_script])

# Chạy prediction script
subprocess.run(["python", prediction_script])
