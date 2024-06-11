import sys
import time

# Общая длина прогресс-бара
TOTAL_BAR_LENGTH = 80
# Время последнего обновления прогресс-бара
LAST_T = time.time()
# Время начала работы прогресс-бара
BEGIN_T = LAST_T


def progress_bar(current, total, msg=None):
    # Отображение прогресса обучения в консоли
    # current(int): текущее значение прогресса
    # total(int): общее количество элементов
    # msg(str, optional): дополнительное сообщение, которое будет отображаться
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()

    # Вычисляем длину заполненной части прогресс-бара
    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    # Выводим прогресс-бар в консоль
    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    # Вычисляем и выводим время, прошедшее с момента последнего обновления и с начала работы
    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    # Функция для форматирования времени в читаемый вид
    # seconds (float): количество секунд
    # Возвращает - str: отформатированное время
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds*1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output
