from enum import Enum

class Models(Enum):
    DEEPSEEK_R1 = 'deepseek/deepseek-r1'
    DEEPSEEK = 'deepseek/deepseek-chat'
    SONNET = 'anthropic/claude-3.5-sonnet:beta'
    O1_PREVIEW = 'openai/o1-preview'
    O1_MINI = 'openai/o1-mini'
    GPT_4O = 'openai/gpt-4o-2024-08-06'
    GPT_4O_MINI = 'openai/gpt-4o-mini'
    GEMINI_FLASH = 'google/gemini-2.0-flash-exp:free'
    MINMAX = 'minimax/minimax-01'
    QWEN = 'qwen/qvq-72b-preview'
    PHI4 = "microsoft/phi-4"

class ClientProvider(Enum):
    OPEN_AI = 'open_ai'
    OPEN_ROUTER = 'open_router'
    OLLAMA = 'ollama'