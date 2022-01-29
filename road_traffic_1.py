import cv2
from tracker import * #importa código com melhor tratamento para rastrear os objetos

# Criar objeto rastreador
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("road_traffic_1.mp4") # carrega o vídeo

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# em branco os objetos de interesse

while True: # para extrair cada quadro do vídeo
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extrair região de interesse
    roi = frame[340: 720,400: 850]

    # Detecção de Objetos
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calcule a área e remova pequenos elementos
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Rastreamento de Objeto
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27: # ao pressionar ESC, sai do loop
        break

cap.release()
cv2.destroyAllWindows()