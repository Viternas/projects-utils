from enum import Enum


from enum import Enum

class Models(Enum):
    DEEPSEEK_R1 = ('deepseek/deepseek-r1', 'open_router')
    DEEPSEEK = ('deepseek/deepseek-chat', 'open_router')
    SONNET = ('anthropic/claude-3.5-sonnet:beta', 'open_router')
    O1_PREVIEW = ('openai/o1-preview', 'openai')
    O1_MINI = ('openai/o1-mini', 'openai')
    GPT_4O = ('openai/gpt-4o-2024-08-06', 'openai')
    GPT_4O_MINI = ('openai/gpt-4o-mini', 'openai')
    GEMINI_FLASH = ('google/gemini-2.0-flash-exp:free', 'open_router')
    MINMAX = ('minimax/minimax-01', 'open_router')
    QWEN = ('qwen/qvq-72b-preview', 'open_router')
    PHI4 = ("microsoft/phi-4", 'open_router')
    DEEPSEEK_R1_14B = ('deepseek-r1:14b', 'ollama')
    GEMMA2_27B = ('gemma2:27b', 'ollama')
    GEMMA2_9B = ('gemma2:9b', 'ollama')
    LLAMA3_2_VISION = ('llama3.2-vision:latest', 'ollama')
    LLAMA3_1_8B = ('llama3.1:8b', 'ollama')
    LLAVA_13B = ('llava:13b', 'ollama')
    LLAVA_7B = ('llava:7b', 'ollama')
    MISTRAL_SMALL_24B = ('mistral-small:24b', 'ollama')
    MISTRAL_NEMO = ('mistral-nemo:latest', 'ollama')
    MISTRAL_7B = ('mistral:7b', 'ollama')
    PHI3_3_8B = ('phi3:3.8b', 'ollama')
    QWEN2_5 = ('qwen2.5:latest', 'ollama')
    MINICPM_V = ('minicpm-v:latest', 'ollama')
    MARCO_O1 = ('marco-o1:latest', 'ollama')
    TULU3 = ('tulu3:latest', 'ollama')

    def __init__(self, model_id, provider):
        self._model_id = model_id
        self.provider = provider

    @property
    def model_id(self):
        return self._model_id


class ClientProvider(Enum):
    OPEN_AI = 'open_ai'
    OPEN_ROUTER = 'open_router'
    OLLAMA = 'ollama'

class ClientProviderEndpoints(Enum):
    OPEN_AI = 'https://api.openai.com/v1'
    Open_Router = 'https://openrouter.ai/api/v1'
    OLLAMA = 'https://chat.kxsb.org/ollama'
