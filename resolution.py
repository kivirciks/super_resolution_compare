from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--input_dir', type=str, required=False,
                        default='C:/Users/n.strokova/Pictures/ITMO/super-resolution/Experiments/small/',
                        help='директория с входными изображениями')
    parser.add_argument('--model', type=str,
                        default='C:/Users/n.strokova/Desktop/super-resolution/model_weights.pth',
                        help='путь к файлу с весами модели')
    parser.add_argument('--output_dir', type=str,
                        default='C:/Users/n.strokova/Desktop/super-resolution/photo/',
                        help='директория для сохранения обработанных изображений')
    args = parser.parse_args()
    print(args)

    # Настройка входных изображений
    GPU_IN_USE = torch.cuda.is_available()
    # Получаем список всех файлов в указанной директории
    file_list = os.listdir(args.input_dir)

    for file_name in file_list:
        # Полный путь к входному файлу
        input_file = os.path.join(args.input_dir, file_name)
        # Открываем изображение и разделяем его на каналы Y, Cb, Cr
        img = Image.open(input_file).convert('YCbCr')
        y, cb, cr = img.split()

        # Импорт и настройка модели
        # Определяем устройство для вычислений
        device = torch.device('cuda' if GPU_IN_USE else 'cpu')
        # Загружаем модель
        model = torch.load(args.model, map_location=lambda storage, loc: storage)
        # Переводим модель на устройство
        model = model.to(device)
        # Преобразуем изображение в тензор
        data = (transforms.ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        # Переводим тензор на устройство
        data = data.to(device)

        # Включаем оптимизацию для CUDA
        if GPU_IN_USE:
            cudnn.benchmark = True

        # Обработка и сохранение изображения
        # Обрабатываем входное изображение с помощью модели
        out = model(data)
        # Переводим результат на CPU
        out = out.cpu()
        # Извлекаем обработанный канал Y
        out_img_y = out.data[0].numpy()
        # Масштабируем значения пикселей
        out_img_y *= 255.0
        # Обрезаем значения вне диапазона [0, 255]
        out_img_y = out_img_y.clip(0, 255)
        # Создаем изображение из канала Y
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        # Увеличиваем размер каналов Cb и Cr с помощью бикубической интерполяции
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

        # Собираем изображение из каналов Y, Cb, Cr и конвертируем в RGB
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        # Полный путь к выходному файлу
        output_file = os.path.join(args.output_dir, file_name)

        # Сохраняем выходное изображение
        out_img.save(output_file)
        print('output image saved to', output_file)


if __name__ == '__main__':
    main()
