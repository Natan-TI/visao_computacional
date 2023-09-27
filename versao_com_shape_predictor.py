import cv2
import dlib

# Carregue o detector de faces da dlib
detector = dlib.get_frontal_face_detector()

# Carregue o modelo para identificação de pontos-chave faciais
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Inicialize a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture um frame da webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converta o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte as faces no frame
    faces = detector(gray)

    # Desenhe retângulos ao redor das faces detectadas
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exiba o número de pessoas detectadas
    num_people = len(faces)
    cv2.putText(frame, f'Pessoas: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exiba o frame resultante
    cv2.imshow('Leitura Facial', frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a webcam e feche a janela da webcam
cap.release()
cv2.destroyAllWindows()
