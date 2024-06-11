import tarfile
from os import remove
from os.path import exists, join, basename
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from .dataset import DatasetFromFolder


def download_bsd300(dest="./dataset"):
    # Загружает и извлекает датасет BSD300 (Berkeley Segmentation Dataset 300)
    # dest (str): путь для сохранения загруженного датасета
    # Возвращает - str: путь к директории с изображениями датасета

    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    # Вычисляет размер кадра, который кратен заданному коэффициенту увеличения.
    # crop_size (int): размер кадра
    # upscale_factor (int): коэффициент увеличения
    # Возвращает - int: размер кадра, кратный заданному коэффициенту увеличения
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    # Создает преобразования для входного изображения
    # crop_size (int): размер кадра
    # upscale_factor (int): коэффициент увеличения
    # Возвращает - Compose: последовательность преобразований для входного изображения

    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    # Создает преобразования для целевого изображения
    # crop_size (int): размер кадра
    # upscale_factor (int): коэффициент увеличения
    # Возвращает - Compose: последовательность преобразований для входного изображения

    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    # Получает датасет для обучения
    # upscale_factor (int): коэффициент увеличения
    # Возвращает - DatasetFromFolder: объект датасета для обучения
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    # Получает датасет для тестирования
    # upscale_factor (int): коэффициент увеличения
    # Возвращает - DatasetFromFolder: объект датасета для тестирования
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))