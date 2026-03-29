# Инструкция по установке и запуску ShadowSpeaker

## Шаг 1: Установка Virtual Audio Cable (Windows)

Для захвата звука из конкретных приложений (Discord, TeamSpeak) нужен виртуальный аудиокабель:

1. Скачайте VB-CABLE: https://vb-audio.com/Cable/
2. Установите и перезагрузите компьютер
3. Настройте маршрутизацию:
   - Откройте "Панель управления" → "Звук"
   - Во вкладке "Воспроизведение" найдите "CABLE Input"
   - В Discord/Teamspeak в настройках звука выберите "CABLE Input" как устройство вывода
   - В Windows в настройках записи выберите "CABLE Output" как устройство по умолчанию

## Шаг 2: Установка Ollama

1. Скачайте Ollama для Windows: https://ollama.com/download
2. Установите и запустите
3. Откройте терминал и скачайте модель:
   ```bash
   ollama pull llama3.2
   ```
   
   Или более легкую модель для GTX 1630:
   ```bash
   ollama pull phi3
   ```

## Шаг 3: Установка Python зависимостей

```bash
cd shadow_speaker
pip install -r requirements.txt
```

### Возможные проблемы и решения:

**Проблема с torch/CUDA:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Проблема с pyannote (требуется токен HuggingFace):**
1. Зарегистрируйтесь на https://huggingface.co
2. Примите условия использования pyannote: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Создайте токен в настройках аккаунта
4. Добавьте переменную окружения:
   ```bash
   set HF_TOKEN=your_token_here
   ```

## Шаг 4: Запуск приложения

```bash
python main.py
```

## Шаг 5: Первое использование

1. При первом запуске приложение проверит доступность компонентов
2. Если все OK - появится сообщение "ShadowSpeaker started - listening for conversations"
3. Начните разговор в Discord/Teamspeak
4. При обнаружении речи собеседника появится оверлей с вариантами ответов
5. Используйте Ctrl+1/2/3 для выбора варианта

## Горячие клавиши

| Клавиши | Действие |
|---------|----------|
| Ctrl+1 | Выбрать вариант ответа №1 |
| Ctrl+2 | Выбрать вариант ответа №2 |
| Ctrl+3 | Выбрать вариант ответа №3 |
| Ctrl+R | Обновить варианты ответов |
| Ctrl+H | Скрыть/показать оверлей |
| Ctrl+Q | Выход из приложения |

## Настройка конфигурации

Отредактируйте `config.py` для изменения параметров:

```python
# Модель Whisper (tiny/base/small/medium/large-v2)
whisper_model = "small"  # быстрее для слабой видеокарты

# Модель LLM
llm_model = "phi3"  # легче чем llama3.2

# Позиция оверлея
overlay_position = "bottom_right"  # top_left, top_right, bottom_left, bottom_right

# Количество вариантов
response_variants_count = 3
```

## Оптимизация для вашей системы

У вас 2 видеокарты: CMP 50HX (майнинговая, без видеовыходов) и GTX 1630 (4GB).

**Рекомендации:**
1. CMP 50HX может использоваться для CUDA вычислений
2. Для GTX 1630 используйте модели поменьше:
   - Whisper: `tiny` или `base`
   - LLM: `phi3`, `tinyllama`, `qwen2:0.5b`

Измените в `config.py`:
```python
whisper_model = "base"
whisper_compute_type = "int8_float16"  # экономия памяти
llm_model = "phi3"
```

## Диагностика проблем

**Проверка аудио устройств:**
```python
import sounddevice as sd
print(sd.query_devices())
```

**Проверка Ollama:**
```bash
curl http://localhost:11434/api/tags
```

**Проверка CUDA:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Логи

Логи сохраняются в папке `logs/shadow_speaker.log`

## Дополнительная помощь

Если возникли проблемы:
1. Проверьте логи в `logs/`
2. Убедитесь что Ollama запущен: `ollama list`
3. Проверьте что Virtual Audio Cable установлен и настроен
4. Убедитесь что у вас достаточно VRAM на видеокарте
