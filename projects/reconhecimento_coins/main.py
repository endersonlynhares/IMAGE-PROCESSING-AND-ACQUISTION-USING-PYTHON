import cv2
import numpy as np

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def pre_process(img):
    img_pre = cv2.GaussianBlur(img, (5, 5), 0)
    img_pre = cv2.medianBlur(img_pre, 13)
    img_pre = cv2.Canny(img_pre, 100, 140)
    img_pre = cv2.dilate(img_pre, np.ones((5, 5), np.uint8), iterations=2)
    img_pre = cv2.erode(img_pre, np.ones((5, 5), np.uint8), iterations=2)
    return img_pre


while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pre = pre_process(img_gray)

    contours, _ = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filtrar contornos pequenos para eliminar ruídos
        if area > 4000:  # Ajuste esse valor conforme necessário
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

            # Calcular o centro do contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Para evitar divisão por zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Escrever o índice dentro do círculo
                cv2.putText(img, str(cnt), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cnt += 1

    cv2.imshow('IMG', img)
    cv2.imshow('PRO', img_pre)
    cv2.waitKey(1)
