import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    """
    Класс для определения сверточной нейронной сети для задачи SR
    Атрибуты:
        relu (nn.ReLU): Активационный слой ReLU.
        conv1 (nn.Conv2d): Первый сверточный слой.
        conv2 (nn.Conv2d): Второй сверточный слой.
        conv3 (nn.Conv2d): Третий сверточный слой.
        conv4 (nn.Conv2d): Четвертый сверточный слой.
        pixel_shuffle (nn.PixelShuffle): Слой перемешивания пикселей.
    """
    def __init__(self, upscale_factor):
        # Инициализирует сеть с заданным коэффициентом масштабирования
        # upscale_factor (int): коэффициент масштабирования для SR
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def _initialize_weights(self):
        # Инициализирует веса сети с использованием ортогональной инициализации
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def forward(self, x):
        # Выполняет прямое распространение входного тензора x через сеть
        # x (torch.Tensor): входной тензор размера (batch_size, 1, height, width)
        # Возвращает - torch.Tensor: выходной тензор размера (batch_size, 1, height * upscale_factor, width * upscale_factor)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x
