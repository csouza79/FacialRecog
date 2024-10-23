import cv2
import dlib
import numpy as np

# Inicializa a detecção de rostos e o detector de pontos faciais de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Baixe o modelo de 68 pontos

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_eye_region(landmarks, start, end):
    points = []
    for i in range(start, end + 1):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return points

def get_iris_position(eye):
    # Encontra o centroide da íris em relação ao olho
    moments = cv2.moments(eye)
    if moments['m00'] == 0:
        return None
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Olho direito (pontos de 36 a 41) e olho esquerdo (pontos de 42 a 47)
        right_eye_points = get_eye_region(landmarks, 36, 41)
        left_eye_points = get_eye_region(landmarks, 42, 47)

        right_eye = np.array(right_eye_points, np.int32)
        left_eye = np.array(left_eye_points, np.int32)

        # Desenhe as regiões dos olhos
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)

        # Calcule a posição da íris
        right_iris = get_iris_position(right_eye)
        left_iris = get_iris_position(left_eye)

        if right_iris:
            cv2.circle(frame, right_iris, 2, (0, 0, 255), -1)
        if left_iris:
            cv2.circle(frame, left_iris, 2, (0, 0, 255), -1)

        # Verifica se a íris está centralizada (aproximadamente)
        # Isso pode variar de acordo com a resolução e calibragem
        if right_iris and left_iris:
            if (right_iris[0] > right_eye_points[0][0] + 5 and
                right_iris[0] < right_eye_points[3][0] - 5 and
                left_iris[0] > left_eye_points[0][0] + 5 and
                left_iris[0] < left_eye_points[3][0] - 5):
                cv2.putText(frame, "Looking at screen", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Not looking at screen", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
