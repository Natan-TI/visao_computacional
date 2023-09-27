import cv2
import dlib

# Carregar o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar um quadro da webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converter o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no quadro
    faces = detector(gray)

    # Contar o número de faces detectadas
    num_faces = len(faces)

    # Desenhar retângulos ao redor das faces detectadas
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibir o número de pessoas detectadas
    cv2.putText(frame, f'Pessoas: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibir o quadro resultante
    cv2.imshow('Detecção Facial', frame)

    # Verificar se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar a janela
cap.release()
cv2.destroyAllWindows()
