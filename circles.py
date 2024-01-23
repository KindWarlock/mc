import random
from math import sqrt
import numpy as np
import cv2

# CAMERA_URL = '/dev/video2'
CAMERA_URL = 'https://192.168.1.39:8080/video'
# CAMERA_URL = 0


def get_camera(url):
    cam = cv2.VideoCapture(url)

    # Камера нашлась?
    if (cam.isOpened() == False):
        print("Error opening video stream or file")
        exit(0)

    # Выставляем параметры
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    return cam


def draw_circles(frame, circles):
    # Проходим по всем найденным кругам и отображаем.
    # Запись circles[0, :] используется, чтобы отбросить первую размерность (по сути, лишнюю); см. слайсы в numpy
    for c in circles[0, :]:
        '''
        Первый параметр - кадр для рисования, второй - координаты центра,
        третий - радиус, четвертый - цвет контура в бгр(!), пятый - толщина контура.        
        '''
        cv2.circle(frame, (c[0], c[1]), c[2], (0, 255, 0), 2)


def find_circles(frame):
    # Переводим из цветного в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Убираем шумы. Чем больше числа параметров - тем больше размытие
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)

    # Для отладки можно использовать следующее:
    # canny = cv2.Canny(blurred, 80, 40)
    # cv2.imshow('Blur', blurred)
    # cv2.imshow('Canny', canny)

    '''
    Находим круги. Первый параметр - изображение для поиска, четвертый - минимальное расстояние между центрами,
    param1 - чем меньше, тем больше контуров будет найдено, param2 - чем выше, тем больше кругов будет найдено.
    '''
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=70, minRadius=10, maxRadius=400)

    # Если ничего не нашли, ничего и возвращаем
    if circles is None:
        return

    # Преобразуем в другой тип для дальнейшей работы
    circles = np.uint16(np.around(circles))
    return circles


def generate_target(frame):
    x = random.randint(0, frame.shape[1])
    y = random.randint(0, frame.shape[0])
    return np.array([x, y])


def check_target_collision(circles, target):
    # Для каждого найденного круга находим расстояние до мишени.
    for c in circles[0, :]:
        dist = sqrt((target['center'][0] - c[0]) ** 2 +
                    (target['center'][1] - c[1]) ** 2)
        if dist < target['radius'] + c[2]:
            return True
    return False


def write_score(frame, score):
    '''
    Кадр, на котором пишем текст, сам текст (строкой!), координаты начала (с левого верхнего угла, (x, y)),
    шрифт, множитель размера шрифта и цвет в BGR
    '''
    cv2.putText(frame, f'Score: {score}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


score = 0
target = {'center': None, 'radius': 20}
cap = get_camera(CAMERA_URL)

# Стандартный цикл при считывании изображения с камеры, можно спокойно копипастить (без внутренней логики, естественно)
while True:
    # ret - считался кадр или нет, frame - кадр
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(
            frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        circles = find_circles(frame)

        if target['center'] is None:
            target['center'] = generate_target(frame)
        cv2.circle(frame, target['center'],
                   target['radius'], (0, 0, 255), 3)
        if circles is not None:
            draw_circles(frame, circles)

            if check_target_collision(circles, target):
                score = score + 1
                print(score)
                target['center'] = None

        write_score(frame, score)

        # Отображаем кадр в окне Stream
        cv2.imshow('Stream', frame)

        # Если в какой-то момент нажали q, закрываемся.
        '''
        cv2.waitKey(1) ждет нажатия указанное время (1 мс), 0&FF - из полученного результата получаем
        только полезную информацию.        
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Сворачиваем лавочку
cap.release()
cv2.destroyAllWindows()
