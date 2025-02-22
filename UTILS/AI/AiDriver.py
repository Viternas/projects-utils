import os
from loguru import logger
from openai import OpenAI, api_key, base_url
from pydantic import BaseModel
from enum import Enum

class ClientProvider(Enum):
    OPEN_AI = 'open_ai'
    OPEN_ROUTER = 'open_router'
    OLLAMA = 'ollama'


class AIDriver:
    def __init__(self,
                 OPEN_ROUTER_KEY: str,
                 OPEN_AI_KEY: str,
                 OLLAMA_KEY: str,
                 AI_MODEL: str = None,
                 agent_prompt: str = None,
                 parser: BaseModel = None):
        logger.debug("Initializing ChatGPT instance.")
        self.open_ai_key = OPEN_AI_KEY
        self.open_router_key = OPEN_ROUTER_KEY
        self.ollama_key = OLLAMA_KEY
        self.client = self.set_client(client_provider=ClientProvider.OPEN_AI)
        self.agent_prompt = agent_prompt
        self.model = AI_MODEL
        self.parser = parser
        logger.info(f"ChatGPT initialized with model: {self.model}")

    def chat_with_gpt(self, prompt):
        logger.info(f"Sending prompt to GPT: {prompt}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.agent_prompt},
                    {"role": "user", "content": prompt},
                ],
                #max_tokens=10000,
                temperature=0.2,
            )
            logger.debug(f"Received response: {response}")
            print(response)
            if response.choices:
                content = response.choices[0].message.content
                logger.success("Response retrieved successfully.")
                return content, response.usage
            else:
                logger.warning("No response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def gpt_parse(self, prompt):
        print(self.model)
        self.model = self.model.split('/')[1]
        #logger.info(f"Parsing prompt with GPT: {prompt[:30]}")
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt},
            ],
            #max_tokens=10000,
            temperature=0.2,
            response_format=self.parser,
        )
        logger.debug(f"Received parse response: {response}")
        if response.choices:
            parsed = response.choices[0].message.parsed
            logger.success("Parsing successful.")
            return parsed, response.usage
        else:
            logger.warning("No parse response from the API.")
            return "No response from the API."

    def open_router_chat(self, prompt):
        self.switch_context(gpt=False)
        logger.info(f"Sending prompt to GPT: {prompt[:50]}")
        try:
            response = self.client.chat.completions.create(
                model=f'{self.model}',
                messages=[
                    {"role": "system", "content": self.agent_prompt},
                    {"role": "user", "content": prompt},
                ],
                #max_tokens=10000,
                temperature=0.1,
            )
            self.switch_context(gpt=True)
            logger.debug(f"Received response: {response}")
            if response.choices:
                content = response.choices[0].message.content
                logger.success("Response retrieved successfully.")
                return content, response.usage
            else:
                logger.warning("No response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def open_parser(self, prompt):
        self.switch_context(gpt=False)
        try:
            response = self.client.beta.chat.completions.parse(
                model=f'openai/{self.model}',
                messages=[
                    {"role": "system", "content": self.agent_prompt},
                    {"role": "user", "content": prompt},
                ],
                #max_tokens=1000,
                temperature=0.2,
                response_format=self.parser,
            )
            logger.debug(f"Received parse response: {response}")
            self.switch_context(gpt=True)
            if response.choices:
                parsed = response.choices[0].message.parsed
                logger.success("Parsing successful.")
                return parsed, response.usage
            else:
                logger.warning("No parse response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while parsing with GPT: {e}")
            return "An error occurred."

    def set_client(self, client_provider: ClientProvider):
        client_mapper = {
            ClientProvider.OPEN_AI: OpenAI(api_key=self.open_ai_key),
            ClientProvider.OPEN_ROUTER: OpenAI(api_key=self.open_router_key, base_url="https://openrouter.ai/api/v1"),
            ClientProvider.OLLAMA: exit
        }
        self.client = client_mapper.get(client_provider)

        if self.client is None:
            raise ValueError(f"Unsupported client provider: {client_provider}")

    def whisper(self, file):
        audio_file = open(file, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text

    def embeddings(self, prompt):
        logger.info(f"Sending prompt to GPT: {prompt}")
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=prompt
            )
            logger.debug(f"Received response: {response}")
            if response.data:
                content = response.data[0].embedding
                logger.success("Response retrieved successfully.")
                return content, response.usage
            else:
                logger.warning("No response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."




if __name__ == "__main__":
    #logger.remove()
    run = AIDriver










