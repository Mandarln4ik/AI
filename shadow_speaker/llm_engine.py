import requests
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from config import Config


class LLMEngine:
    """Движок для работы с различными LLM провайдерами"""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm.provider
        self.model_name = config.llm.model_name
        self.base_url = config.llm.base_url
        self.lmstudio_port = config.llm.lmstudio_port
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.context_length = config.llm.context_length
        
        # История запросов для отладки
        self.request_history: List[Dict[str, Any]] = []
        
        # Проверка доступности провайдера
        self._check_provider_availability()
    
    def _check_provider_availability(self) -> bool:
        """Проверка доступности провайдера"""
        try:
            if self.provider == "ollama":
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
            elif self.provider == "lmstudio":
                response = requests.get(f"http://localhost:{self.lmstudio_port}/v1/models", timeout=5)
                return response.status_code == 200
            elif self.provider == "llama_cpp":
                # llama.cpp обычно работает через локальный сервер, проверяем как ollama
                return True  # Предполагаем доступность
            else:
                print(f"Неизвестный провайдер: {self.provider}")
                return False
        except Exception as e:
            print(f"Провайдер {self.provider} недоступен: {e}")
            return False
    
    def generate_response(self, 
                         dialogue_context: str, 
                         screen_context: str = "", 
                         screenshot_base64: Optional[str] = None,
                         num_options: int = 3) -> List[str]:
        """
        Генерация вариантов ответов на основе контекста диалога и экрана
        
        Args:
            dialogue_context: Текст диалога за последние несколько минут
            screen_context: Описание визуального контекста
            screenshot_base64: Скриншот в base64 (для мультимодальных моделей)
            num_options: Количество вариантов ответов для генерации
        
        Returns:
            Список вариантов ответов
        """
        system_prompt = """Ты - умный ассистент, который помогает пользователю в реальных диалогах.
Твоя задача - анализировать контекст беседы и предлагать 3 варианта ответа, которые пользователь может использовать.

Требования к ответам:
1. Ответы должны быть естественными и соответствовать стилю беседы
2. Учитывай контекст диалога (что обсуждалось ранее)
3. Если есть визуальный контекст (игра, приложение), учитывай его
4. Предложи разные варианты: согласиться, возразить, уточнить, перевести тему
5. Ответы должны быть краткими (1-2 предложения)
6. Язык ответов должен соответствовать языку диалога

Формат вывода:
Просто перечисли 3 варианта, каждый с новой строки, без нумерации и дополнительных пояснений."""

        user_prompt = f"""Контекст диалога (последние 5 минут):
{dialogue_context}

{screen_context}

Сгенерируй 3 варианта ответа для пользователя:"""

        try:
            if self.provider == "ollama":
                return self._generate_ollama(system_prompt, user_prompt, num_options)
            elif self.provider == "lmstudio":
                return self._generate_lmstudio(system_prompt, user_prompt, num_options)
            elif self.provider == "llama_cpp":
                return self._generate_llama_cpp(system_prompt, user_prompt, num_options)
            else:
                raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
        
        except Exception as e:
            print(f"Ошибка генерации ответа: {e}")
            return ["Извините, не могу сейчас помочь с ответом.", 
                    "Давайте продолжим разговор позже.",
                    "Я анализирую ситуацию..."]
    
    def _generate_ollama(self, system_prompt: str, user_prompt: str, num_options: int) -> List[str]:
        """Генерация через Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("response", "")
        
        # Парсинг вариантов ответов
        options = self._parse_response_options(generated_text, num_options)
        
        self._log_request("ollama", payload, options)
        
        return options
    
    def _generate_lmstudio(self, system_prompt: str, user_prompt: str, num_options: int) -> List[str]:
        """Генерация через LM Studio (OpenAI-compatible API)"""
        url = f"http://localhost:{self.lmstudio_port}/v1/chat/completions"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]
        
        options = self._parse_response_options(generated_text, num_options)
        
        self._log_request("lmstudio", payload, options)
        
        return options
    
    def _generate_llama_cpp(self, system_prompt: str, user_prompt: str, num_options: int) -> List[str]:
        """Генерация через llama.cpp (через локальный сервер)"""
        # По умолчанию используем тот же API что и Ollama, если сервер запущен
        # Можно настроить под конкретную реализацию llama.cpp
        return self._generate_ollama(system_prompt, user_prompt, num_options)
    
    def _parse_response_options(self, text: str, num_options: int) -> List[str]:
        """Парсинг сгенерированного текста в список вариантов"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Фильтрация пустых строк и служебных символов
        options = []
        for line in lines:
            # Удаление нумерации (1., 2., 3. или - или *)
            cleaned = line.lstrip('0123456789.-* ')
            if cleaned and len(cleaned) > 5:  # Минимальная длина ответа
                options.append(cleaned)
        
        # Если не хватило вариантов, дополняем заглушками
        while len(options) < num_options:
            options.append("Нужно подумать над ответом...")
        
        return options[:num_options]
    
    def _log_request(self, provider: str, request_data: Dict, response_options: List[str]) -> None:
        """Логирование запроса для отладки"""
        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "request": request_data,
            "response_count": len(response_options)
        })
        
        # Ограничение истории
        if len(self.request_history) > 100:
            self.request_history.pop(0)
    
    def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей от провайдера"""
        try:
            if self.provider == "ollama":
                response = requests.get(f"{self.base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
            
            elif self.provider == "lmstudio":
                response = requests.get(f"http://localhost:{self.lmstudio_port}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [model["id"] for model in data.get("data", [])]
            
            elif self.provider == "llama_cpp":
                # Для llama.cpp список моделей зависит от реализации
                return [self.model_name]
        
        except Exception as e:
            print(f"Ошибка получения списка моделей: {e}")
        
        return [self.model_name]
    
    def test_connection(self) -> Dict[str, Any]:
        """Тестирование соединения с провайдером"""
        result = {
            "success": False,
            "provider": self.provider,
            "model": self.model_name,
            "message": ""
        }
        
        try:
            if self.provider == "ollama":
                response = requests.get(f"{self.base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    result["success"] = True
                    result["message"] = "Ollama подключена успешно"
                else:
                    result["message"] = f"Ошибка подключения к Ollama: {response.status_code}"
            
            elif self.provider == "lmstudio":
                response = requests.get(f"http://localhost:{self.lmstudio_port}/v1/models", timeout=10)
                if response.status_code == 200:
                    result["success"] = True
                    result["message"] = "LM Studio подключена успешно"
                else:
                    result["message"] = f"Ошибка подключения к LM Studio: {response.status_code}"
            
            elif self.provider == "llama_cpp":
                result["success"] = True
                result["message"] = "llama.cpp настроен (требуется ручной запуск сервера)"
            
            else:
                result["message"] = f"Неизвестный провайдер: {self.provider}"
        
        except Exception as e:
            result["message"] = f"Ошибка подключения: {str(e)}"
        
        return result
