## Описание репозитория

Репозиторий [hse-dl-asr-project](https://github.com/asvistunov/hse-dl-asr-project) представляет собой проект по задаче автоматического распознавания речи (ASR) в рамках курса «Глубокое обучение в аудио» (Deep Learning in Audio, DLA) в НИУ ВШЭ.

Данный репозиторий содержит готовое решение с обученной моделью, которое можно использовать для инференса и дальнейшего анализа. В проекте реализованы все необходимые компоненты, включая скрипты для обучения и инференса, конфигурационные файлы и базовую структуру проекта.

### Отчет о выполнении

Для подробного отчета о ходе выполнения проекта можно ознакомиться с отчетом на Weights & Biases:  
[Ссылка на WB отчет](https://wandb.ai/coolduck-hse/pytorch_template_asr_example/reports/-DeepSpeech---VmlldzoxMDgzMzgwNA?accessToken=93eeonqqtp9r1jk8jw7lwwbh9t60miebfwllyv6nufl30czayvl549mow5c8urue)

### Установка проекта

1. **Клонируйте репозиторий**:

   ```bash
   git clone https://github.com/asvistunov/hse-dl-asr-project.git
   cd hse-dl-asr-project
   ```

2. **Создайте и активируйте виртуальное окружение** (рекомендуется):

   - С использованием `venv`:

     ```bash
     python3 -m venv project_env
     source project_env/bin/activate
     ```

   - С использованием `conda`:

     ```bash
     conda create -n project_env python=3.10
     conda activate project_env
     ```

3. **Установите необходимые зависимости**:

   ```bash
   pip install -r requirements.txt
   ```

---

### Запуск инференса

Для запуска инференса с обученной моделью выполните:

```bash
gdown 1Cu3ke8Zfc6_oQXzw-fPP2fW1ZJVc0boP -O saved.zip
unzip saved.zip -d .
python3 inference.py
```

Скрипт автоматически загрузит обученную модель и выполнит распознавание речи на тестовых данных.

---

### Запуск обучения.

Если требуется обучение модели с нуля, выполните следующую команду:

```bash
python3 train.py
```

---

*Примечание: данный проект распространяется под лицензией MIT.*
