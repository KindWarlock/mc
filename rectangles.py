import numpy as np
import cv2

# Цвета HSV-маски, по которой будем искать объекты. Берем что-то из синих оттенков.
hsv_mask = np.array([[110, 20, 35], [180, 255, 255]])


def preprocess(img):
    result = cv2.GaussianBlur(img, (3, 3), 1)
    return result


def find_by_mask(img, mask_color):
    # Переводим изображение в hsv-формат
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Создаем маску наподобие нашего изображения; пока пустую
    mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype="uint8")

    # Добавляем элементы заданных оттенков в маску
    mask += cv2.inRange(hsv, mask_color[0], mask_color[1])

    # Накладываем маску
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('gray', blur)

    # Выделение контуров на изображении
    canny = cv2.Canny(blur, 40, 150)
    cv2.imshow('Canny', canny)

    # Находим контуры (по сути, преобразуем полученные в нужный тип)
    contours, hierarchy = cv2.findContours(canny, 1, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры. Третий параметр - номер контура в массиве, если -1 - рисуются все
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    return contours


def find_rectangle(img, contours):
    for cnt in contours:
        # Спасибо OpenCV за миллион ненужных размерностей
        x1, y1 = cnt[0][0]

        # Находим приблизительную форму
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

        # Отделяем четырехугольники
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)

            # Костыль для удаления шума
            if w > 10:
                print(x, y, w, h)
                img = cv2.drawContours(img, [cnt], -1, (0, 0, 255), 3)


img = cv2.imread('books.jpg')
img = cv2.resize(
    img, (img.shape[1] // 2, img.shape[0] // 2))
img = preprocess(img)


masked = find_by_mask(img, hsv_mask)

# Для выделения всех прямоугольников: masked = img
contours = find_contours(masked)
find_rectangle(masked, contours)

cv2.imshow("Book?", masked)

# Останавливаем выполнение, пока не будет нажата клавиша
cv2.waitKey(0)
cv2.destroyAllWindows()
