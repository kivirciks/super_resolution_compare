from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SubPixelCNN.model import Net
from progress_bar import progress_bar


class SubPixelTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SubPixelTrainer, self).__init__()
        # Проверяем, доступна ли CUDA
        self.CUDA = torch.cuda.is_available()
        # Определяем устройство для модели
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        # Инициализируем модель
        self.model = None
        # Скорость обучения
        self.lr = config.lr
        # Количество эпох обучения
        self.nEpochs = config.nEpochs
        # Функция потерь
        self.criterion = None
        # Оптимизатор
        self.optimizer = None
        # Планировщик скорости обучения
        self.scheduler = None
        # Случайное зерно
        self.seed = config.seed
        # Коэффициент масштабирования
        self.upscale_factor = config.upscale_factor
        # Загрузчик тренировочных данных
        self.training_loader = training_loader
        # Загрузчик тестовых данных
        self.testing_loader = testing_loader

    def build_model(self):
        # Создает модель SUB и настраивает компоненты обучения
        self.model = Net(upscale_factor=self.upscale_factor).to(self.device)
        # Определяем функцию потерь
        self.criterion = torch.nn.MSELoss()
        # Устанавливаем случайное зерно
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            # Включаем оптимизацию CUDA
            cudnn.benchmark = True
            # Переносим функцию потерь на GPU
            self.criterion.cuda()

        # Определяем оптимизатор
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Определяем планировщик скорости обучения
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    def save(self):
        # Сохраняет обученную модель в файл
        model_out_path = "SUB.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        # Обучение модели SUB
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # Переносим данные на устройство
            data, target = data.to(self.device), target.to(self.device)
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            # Вычисляем потери
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            # Вычисляем градиенты
            loss.backward()
            # Обновляем параметры модели
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        # Оценивает модель SRCNN на тестовом наборе
        # Переводим модель в режим оценки
        self.model.eval()
        avg_psnr = 0

        # Отключаем вычисление градиентов
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                # Переносим данные на устройство
                data, target = data.to(self.device), target.to(self.device)
                # Получаем предсказание модели
                prediction = self.model(data)
                # Вычисляем среднеквадратичную ошибку
                mse = self.criterion(prediction, target)
                # Вычисляем PSNR
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        # Запускает цикл обучения и тестирования в течение указанного числа эпох
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()
