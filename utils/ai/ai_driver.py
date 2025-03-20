import os
import json
from loguru import logger
from openai import OpenAI, api_key, base_url
from ollama import Client as OllamaClient
from pydantic import BaseModel
from utils.ai.ai_enums import *
from utils.ai.ai_basemodels import *

class AIDriver:
    def __init__(self,
                 OPEN_ROUTER_KEY: str,
                 OPEN_AI_KEY: str,
                 OLLAMA_KEY: str,
                 AI_MODEL: str = None,
                 agent_prompt: str = None,
                 parser: BaseModel = None):
        logger.info("Initializing AiDriver instance.")
        self.open_ai_key = OPEN_AI_KEY
        self.open_router_key = OPEN_ROUTER_KEY
        self.ollama_key = OLLAMA_KEY
        self.agent_prompt = agent_prompt
        if self.agent_prompt is None:
            self.agent_prompt = ''
        self.model = AI_MODEL
        self.parser = parser
        self.client = None
        logger.success(f"AiDriver initialized")

    def open_ai_chat(self, prompt):
        logger.info(f"Sending prompt to GPT: {prompt}")
        self.set_client(client_provider=ClientProvider.OPEN_AI)
        try:
            response_params = {
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt},
            }
            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                response_params = {
                    {"role": "user", "content": self.agent_prompt + prompt},
                }
            self.model = self.model.split('/')[1]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=response_params,
                # max_tokens=10000,
                #temperature=0.2,
            )
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

    def gpt_parse(self, prompt: str):
        self.set_client(client_provider=ClientProvider.OPEN_AI)
        if not self.parser:
            logger.error('No parser set')
            return False
        try:
            logger.info(f"Parsing prompt with GPT: {prompt[:30]}")
            response_params = {
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt},
            }
            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                response_params = {
                    {"role": "user", "content": self.agent_prompt + prompt},
                }
            self.model = self.model.split('/')[1]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[response_params],
                respone_format=self.parser
                # max_tokens=10000,
                # temperature=0.2,
            )
            logger.debug(f"Received parse response: {response}")
            if response.choices:
                parsed = response.choices[0].message.parsed
                logger.success("Parsing successful.")
                return parsed, response.usage
            else:
                logger.warning("No parse response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def open_router_chat(self, prompt: str) -> tuple[str, dict] | str:
        """
        Send a chat completion request through OpenRouter API.

        Args:
            prompt (str): The user's input prompt

        Returns:
            tuple[str, dict]: Tuple containing response content and usage statistics
            str: Error message if the request fails
        """
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}. Expected str.")
            return "Invalid prompt type. Expected string."

        if not prompt.strip():
            logger.warning("Empty prompt received.")
            return "Empty prompt received."

        if not self.model:
            logger.error("No model specified for OpenRouter chat.")
            return "No model specified."

        self.set_client(client_provider=ClientProvider.OPEN_ROUTER)
        logger.info(f"Sending prompt to OpenRouter: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.agent_prompt or ""},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                # Additional parameters can be added here as needed
                # max_tokens=10000,  # Uncomment and adjust if needed
            )

            logger.debug(f"Received response: {response}")

            if not response:
                logger.error("Received empty response from OpenRouter")
                return "Empty response received from API."

            if not hasattr(response, 'choices'):
                logger.error("Invalid response format: missing 'choices' attribute")
                return "Invalid response format from API."

            if not response.choices:
                logger.warning("No choices in response from OpenRouter")
                return "No response choices available."

            content = response.choices[0].message.content
            usage_stats = getattr(response, 'usage', {})

            if not content:
                logger.warning("Empty content in response")
                return "Empty content in response."

            logger.success("Response retrieved successfully")
            return content, usage_stats

        except ConnectionError as e:
            logger.error(f"Connection error while calling OpenRouter: {e}")
            return f"Connection error: {str(e)}"

        except TimeoutError as e:
            logger.error(f"Timeout while calling OpenRouter: {e}")
            return f"Request timed out: {str(e)}"

        except Exception as e:
            logger.error(f"Error during OpenRouter chat completion: {e}")
            return f"An error occurred: {str(e)}"

    def open_parser(self, prompt):
        self.set_client(ClientProvider.OPEN_ROUTER)
        try:
            response = self.client.beta.chat.completions.parse(
                model=f'{self.model}',
                messages=[
                    {"role": "system", "content": self.agent_prompt},
                    {"role": "user", "content": prompt},
                ],

                response_format=self.parser.model_json_schema(),
            )
            logger.debug(f"Received parse response: {response}")
            if response.choices:
                parsed = response.choices[0].message.content
                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success("Parsing successful.")
                    return result, response.usage
                except:
                    logger.error("Parsing Failed, moving to default.")
                    self.model = Models.O3_MINI.model_id
                    return self.gpt_parse(prompt=parsed)
            else:
                logger.warning("No parse response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def ollama_chat(self, prompt):
        logger.info(f"Sending prompt to Ollama: {prompt}")
        self.set_client(client_provider=ClientProvider.OLLAMA)
        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": self.agent_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            logger.debug(f"Received response: {response}")
            if response:
                content = response.response
                usage = convert_ollama_to_openai_usage(ollama_output=response.dict())
                logger.success("Response retrieved successfully.")
                """
                Need to standard the usage return to be the same type as openai standards
                """
                return content, usage
            else:
                logger.warning("No response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def ollama_parser(self, prompt):
        self.set_client(ClientProvider.OLLAMA)
        try:
            response = self.client.chat(
                      messages=[
                          {"role": "system", "content": self.agent_prompt},
                          {"role": "user", "content": prompt},
                      ],
                      model=self.model,
                      format=self.parser.model_json_schema(),
                    )
            logger.debug(f"Received parse response: {response}")
            if response:
                parsed = response.message.content
                usage = convert_ollama_to_openai_usage(ollama_output=response.dict())
                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success("Parsing successful.")
                    return result, usage
                except:
                    logger.error("Parsing Failed, moving to default.")
                    self.model = Models.O3_MINI.model_id
                    return self.gpt_parse(prompt=parsed)
            else:
                logger.warning("No parse response from the API.")
                return "No response from the API."
        except Exception as e:
            logger.error(f"An error occurred while chatting with GPT: {e}")
            return "An error occurred."

    def whisper(self, file_path: str) -> tuple[str, dict] | str:
        """
        Transcribe audio using OpenAI's Whisper model.

        Args:
            file_path (str): Path to the audio file to transcribe

        Returns:
            tuple[str, dict]: Tuple containing transcription text and usage stats
            str: Error message if transcription fails
        """
        logger.info(f"Transcribing audio file: {file_path}")
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return "Audio file not found."

            # Verify file is readable
            if not os.access(file_path, os.R_OK):
                logger.error(f"Audio file not readable: {file_path}")
                return "Audio file not readable."

            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            logger.debug(f"Received transcription response: {transcription}")

            if hasattr(transcription, 'text'):
                logger.success("Transcription completed successfully.")
                return transcription.text, getattr(transcription, 'usage', {})
            else:
                logger.warning("No transcription text in response.")
                return "No transcription text in response."

        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}")
            return f"An error occurred during transcription: {str(e)}"

    def embeddings(self, prompt):
        logger.info(f"Sending prompt to GPT: {prompt}")
        self.set_client(ClientProvider.OPEN_AI)
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

    def set_client(self, client_provider: ClientProvider):
        client_mapper = {
            ClientProvider.OPEN_AI: lambda: OpenAI(
                api_key=self.open_ai_key,
                base_url=ClientProviderEndpoints.OPEN_AI.value),
            ClientProvider.OPEN_ROUTER: lambda: OpenAI(
                api_key=self.open_router_key,
                base_url=ClientProviderEndpoints.Open_Router.value),
            ClientProvider.OLLAMA: lambda: OllamaClient(
                headers={"Authorization": f"Bearer {self.ollama_key}"},
                host=ClientProviderEndpoints.OLLAMA.value)
        }

        client_factory = client_mapper.get(client_provider)
        logger.info(f'Setting client to {client_provider}')
        if client_factory is None:
            raise ValueError(f"Unsupported client provider: {client_provider}")

        self.client = client_factory()
        logger.success(f'Client successfully set to {client_provider}')


if __name__ == "__main__":
    # logger.remove()
    run = AIDriver
