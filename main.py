
# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np

 
def eye_aspect_ratio(eye):
    # Calcula a relação de aspecto do olho para detectar piscadas
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_looking_forward(shape):
    # Obtém os pontos dos olhos (conjunto de landmarks 36-41 para o olho esquerdo e 42-47 para o direito)
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # Calcula o centro de cada olho
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    # Define limites para considerar que o usuário está olhando para frente
    # Aqui, assumimos que se a posição horizontal do centro dos olhos está próximo ao centro dos olhos,
    # o usuário está olhando para frente.
    threshold = 0.2  # Ajuste fino dessa tolerância conforme necessário

    # Cálculo da relação pupila-canto (a posição do centro do olho em relação aos extremos)
    left_ratio = (left_eye_center[0] - left_eye[0][0]) / (left_eye[3][0] - left_eye[0][0])
    right_ratio = (right_eye_center[0] - right_eye[0][0]) / (right_eye[3][0] - right_eye[0][0])

    # Verifica se ambos os olhos estão centrados
    if abs(left_ratio - 0.5) < threshold and abs(right_ratio - 0.5) < threshold:
        return True  # Está olhando para a câmera
    return False


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

         # Verifica se o usuário está olhando para frente
        if is_looking_forward(shape):
            cv2.putText(image, "Looking Forward", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Not Looking Forward", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
