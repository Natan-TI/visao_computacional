import cv2

# Inicialize o classificador de detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialize o contador de rostos detectados
contador_de_rostos = 0

# Inicialize a webcamq
cap = cv2.VideoCapture(0)

while True:
    # Capture um quadro da webcam
    ret, frame = cap.read()

    # Converta a imagem para escala de cinza (a detecção de rosto funciona melhor em imagens em escala de cinza)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Atualize o contador de rostos detectados
    contador_de_rostos = len(faces)

    # Desenhe um retângulo em torno de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Adicione o número de pessoas detectadas na imagem
    cv2.putText(frame, f'Pessoas: {contador_de_rostos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exiba o quadro com as detecções
    cv2.imshow('Detecção de Rosto', frame)

    # Saia do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libere a webcam e feche a janela
cap.release()
cv2.destroyAllWindows()

# Exiba o número total de pessoas detectadas
print(f"Total de pessoas detectadas: {contador_de_rostos}")