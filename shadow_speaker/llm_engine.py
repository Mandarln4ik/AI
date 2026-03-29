"""
LLM движок для генерации вариантов ответов
Поддержка Ollama, LM Studio и llama.cpp
"""
import requests
from typing import List, Dict, Optional
from loguru import logger
import json
import threading


class LLMEngine:
    """
    Движок для генерации ответов через локальную LLM
    Поддерживаемые провайдеры: ollama, lmstudio, llama_cpp
    """
    
    def __init__(self, config):
        self.config = config
        self.is_initialized = False
        self._generation_lock = threading.Lock()
        self.provider_name = "Unknown"
        
        # Контекст беседы
        self.conversation_history = []
        
        # Системный промпт
        self.system_prompt = """Ты - умный ассистент который помогает пользователю в диалогах.
Твоя задача - предлагать краткие, естественные варианты ответов для использования в разговоре.

Правила:
1. Ответы должны быть короткими (1-2 предложения)
2. Естественными и подходящими для устной речи
3. Предложи 3 разных варианта с разной тональностью:
   - Нейтральный/вежливый
   - Дружелюбный/неформальный  
   - Краткий/по делу
4. Учитывай контекст беседы
5. Не используй эмодзи и специальные символы
6. Пиши на том же языке что и собеседник

Формат ответа строго JSON:
{
  "variants": [
    {"style": "neutral", "text": "..."},
    {"style": "friendly", "text": "..."},
    {"style": "concise", "text": "..."}
  ]
}"""
    
    def initialize(self) -> bool:
        """Инициализация соединения с LLM"""
        try:
            if self.config.llm_provider == "ollama":
                # Проверка доступности Ollama
                response = requests.get(f"{self.config.llm_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    if self.config.llm_model in model_names or any(self.config.llm_model in m for m in model_names):
                        logger.info(f"Ollama is running with model: {self.config.llm_model}")
                    else:
                        logger.warning(f"Model {self.config.llm_model} not found. Available: {model_names}")
                        logger.warning("Please run: ollama pull " + self.config.llm_model)
                    
                    self.provider_name = "Ollama"
                    self.is_initialized = True
                    return True
                else:
                    logger.error(f"Ollama not responding: {response.status_code}")
                    logger.error("Please start Ollama service")
                    return False
                    
            elif self.config.llm_provider == "lmstudio":
                # Проверка доступности LM Studio Server
                response = requests.get(f"{self.config.llm_host}/v1/models", timeout=5)
                if response.status_code == 200:
                    models_data = response.json().get("data", [])
                    model_names = [m["id"] for m in models_data]
                    
                    if self.config.llm_model in model_names or any(self.config.llm_model in m for m in model_names):
                        logger.info(f"LM Studio is running with model: {self.config.llm_model}")
                    else:
                        logger.warning(f"Model {self.config.llm_model} not found in LM Studio. Available: {model_names}")
                        logger.warning("Make sure to load a model in LM Studio and enable local server")
                    
                    self.provider_name = "LM Studio"
                    self.is_initialized = True
                    return True
                else:
                    logger.error(f"LM Studio not responding: {response.status_code}")
                    logger.error("Please start LM Studio and enable 'Local Server' on port 1234")
                    return False
            
            elif self.config.llm_provider == "llama_cpp":
                # llama.cpp через прямой вызов (требует установки llama-cpp-python)
                try:
                    from llama_cpp import Llama
                    logger.info(f"llama_cpp provider selected, will load model: {self.config.llm_model}")
                    self.provider_name = "llama.cpp"
                    self.is_initialized = True
                    return True
                except ImportError:
                    logger.error("llama-cpp-python not installed")
                    logger.error("Install with: pip install llama-cpp-python")
                    return False
            
            else:
                logger.error(f"Unsupported LLM provider: {self.config.llm_provider}")
                logger.error("Supported providers: ollama, lmstudio, llama_cpp")
                return False
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to LLM provider: {e}")
            if self.config.llm_provider == "ollama":
                logger.error("Start with: ollama serve")
            elif self.config.llm_provider == "lmstudio":
                logger.error("Open LM Studio and enable 'Local Server' in settings")
            return False
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return False
    
    def generate_response_variants(self, conversation_context: str, user_speaker: str, screen_context: str = "") -> List[Dict[str, str]]:
        """
        Генерация вариантов ответов на основе контекста беседы
        
        Args:
            conversation_context: Текст последних реплик
            user_speaker: ID спикера пользователя
            screen_context: Визуальный контекст с экрана (опционально)
        
        Returns:
            Список вариантов ответов
        """
        if not self.is_initialized:
            logger.warning("LLM not initialized")
            return self._get_fallback_responses()
        
        with self._generation_lock:
            try:
                # Формируем промпт с учетом визуального контекста
                prompt = self._build_prompt(conversation_context, user_speaker, screen_context)
                
                if self.config.llm_provider == "ollama":
                    variants = self._query_ollama(prompt)
                    if variants:
                        return variants
                
                elif self.config.llm_provider == "lmstudio":
                    variants = self._query_lmstudio(prompt)
                    if variants:
                        return variants
                
                elif self.config.llm_provider == "llama_cpp":
                    variants = self._query_llama_cpp(prompt)
                    if variants:
                        return variants
                
                return self._get_fallback_responses()
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                return self._get_fallback_responses()
    
    def _build_prompt(self, context: str, user_speaker: str, screen_context: str = "") -> str:
        """Построение промпта для LLM с учетом визуального контекста"""
        # Определяем последнюю реплику не от пользователя
        lines = context.split("\n")
        last_other_speech = ""
        
        for line in reversed(lines):
            if line.strip() and not line.startswith(user_speaker):
                last_other_speech = line
                break
        
        prompt = f"""Контекст беседы:
{context}

Последняя реплика собеседника: {last_other_speech}
"""
        
        if screen_context:
            prompt += f"\nВизуальный контекст (что видно на экране): {screen_context}\n"
        
        prompt += f"\nПредложи 3 варианта ответа для {user_speaker}.\n"
        return prompt
    
    def _query_ollama(self, prompt: str) -> Optional[List[Dict[str, str]]]:
        """Запрос к Ollama API"""
        try:
            payload = {
                "model": self.config.llm_model,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.llm_temperature,
                    "num_predict": self.config.llm_max_tokens,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.config.llm_host}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                
                # Парсим JSON из ответа
                variants = self._parse_response_variants(text)
                if variants:
                    logger.info(f"Generated {len(variants)} response variants via Ollama")
                    return variants
            
            logger.warning(f"Ollama returned status {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return None
    
    def _query_lmstudio(self, prompt: str) -> Optional[List[Dict[str, str]]]:
        """Запрос к LM Studio API (OpenAI-compatible)"""
        try:
            payload = {
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.config.llm_host}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Парсим JSON из ответа
                variants = self._parse_response_variants(text)
                if variants:
                    logger.info(f"Generated {len(variants)} response variants via LM Studio")
                    return variants
            
            logger.warning(f"LM Studio returned status {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"LM Studio query failed: {e}")
            return None
    
    def _query_llama_cpp(self, prompt: str) -> Optional[List[Dict[str, str]]]:
        """Запрос к llama.cpp напрямую"""
        try:
            from llama_cpp import Llama
            
            # Загружаем модель если путь указан
            model_path = self.config.llm_model
            if not model_path.endswith('.gguf'):
                logger.error("llama_cpp requires a .gguf model file path")
                return None
            
            llm = Llama(
                model_path=model_path,
                n_ctx=self.config.llm_context_length,
                n_gpu_layers=-1,  # Использовать все слои GPU
                verbose=False
            )
            
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            output = llm(
                full_prompt,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                top_p=0.9,
                stop=["}", "\n\n"],
                echo=False
            )
            
            text = output["choices"][0]["text"]
            
            # Парсим JSON из ответа
            variants = self._parse_response_variants(text)
            if variants:
                logger.info(f"Generated {len(variants)} response variants via llama.cpp")
                return variants
            
            return None
            
        except Exception as e:
            logger.error(f"llama.cpp query failed: {e}")
            return None
    
    def _parse_response_variants(self, text: str) -> Optional[List[Dict[str, str]]]:
        """Парсинг JSON ответа с вариантами"""
        try:
            # Ищем JSON в тексте
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                data = json.loads(json_str)
                
                if "variants" in data:
                    variants = data["variants"]
                    # Валидация
                    if isinstance(variants, list) and len(variants) >= 1:
                        return variants[:self.config.response_variants_count]
            
            # Пробуем найти варианты в другом формате
            lines = text.strip().split("\n")
            variants = []
            
            for i, line in enumerate(lines[:5], 1):
                clean_line = line.strip().lstrip("123456789.-)").strip()
                if clean_line and len(clean_line) > 5:
                    variants.append({
                        "style": f"variant_{i}",
                        "text": clean_line
                    })
            
            if variants:
                return variants[:self.config.response_variants_count]
            
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing variants: {e}")
            return None
    
    def _get_fallback_responses(self) -> List[Dict[str, str]]:
        """Запасные варианты если LLM недоступна"""
        return [
            {"style": "neutral", "text": "Понял, продолжайте."},
            {"style": "friendly", "text": "Интересно! Расскажи подробнее."},
            {"style": "concise", "text": "Ясно."}
        ]
    
    def add_to_history(self, speaker: str, text: str):
        """Добавить реплику в историю"""
        self.conversation_history.append({
            "speaker": speaker,
            "text": text,
            "timestamp": None  # можно добавить время
        })
        
        # Ограничиваем историю
        max_history = 20
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def clear_history(self):
        """Очистить историю беседы"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
