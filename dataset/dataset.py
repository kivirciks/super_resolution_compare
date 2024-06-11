from os import listdir
from os.path import join
import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    # проверка, является ли расширение файла поддерживаемым
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    # загружает изображение из указанного пути и возвращает только канал Y (яркость) в формате YCbCr
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    # Инициализация класса DatasetFromFolder
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        # image_dir (str): Путь к директории с изображениями
        # input_transform (callable, optional): преобразование входного изображения
        # target_transform (callable, optional): преобразование целевого изображения

        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # Получение элемента из датасета по индексу
        # index (int): индекс элемента
        # возвращает: кортеж, содержащий входное изображение и целевое изображение

        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        # возвращает длину датасета (кол-во изображений)
        return len(self.image_filenames)