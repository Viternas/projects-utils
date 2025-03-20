import os
import json
import time
from loguru import logger
from openai import OpenAI, api_key, base_url
from ollama import Client as OllamaClient
from pydantic import BaseModel
from utils.ai.ai_enums import *
from utils.ai.ai_basemodels import *
from typing import Callable, Any, Tuple, Optional, Dict, Union


class AIDriver:
    def __init__(self,
                 OPEN_ROUTER_KEY: str,
                 OPEN_AI_KEY: str,
                 OLLAMA_KEY: str,
                 AI_MODEL: str = None,
                 agent_prompt: str = None,
                 parser: BaseModel = None):
        """Initialize AI Driver with appropriate API keys and configurations."""
        logger.info("Initializing AIDriver with model: {}", AI_MODEL or "None")
        self.open_ai_key = OPEN_AI_KEY
        self.open_router_key = OPEN_ROUTER_KEY
        self.ollama_key = OLLAMA_KEY
        self.agent_prompt = agent_prompt or ''
        self.model = AI_MODEL
        self.parser = parser
        self.client = None

        # Log configuration details without sensitive information
        logger.debug("AIDriver configuration: model={}, parser={}, agent_prompt_length={}",
                     self.model,
                     self.parser.__class__.__name__ if self.parser else None,
                     len(self.agent_prompt))
        logger.success("AIDriver initialized successfully")

    def _truncate_text(self, text: str, max_length: int = 50) -> str:
        """Helper method to truncate text for logging purposes."""
        if not text or len(text) <= max_length:
            return text
        return f"{text[:max_length]}..."

    def _log_prompt(self, prompt: str, provider: str) -> None:
        """Standardized prompt logging with truncation."""
        truncated = self._truncate_text(prompt)
        logger.info("Sending prompt to {}: {}", provider, truncated)
        logger.debug("Full prompt length: {}", len(prompt))

    def _log_response_received(self, response: Any, provider: str) -> None:
        """Log receipt of response with limited information."""
        # Log minimal information at info level
        logger.info("Received response from {}", provider)
        # Log more details at debug level
        if hasattr(response, 'usage'):
            logger.debug("Response usage stats: {}", response.usage)
        logger.trace("Full response object: {}", response)  # Use trace for very verbose logs

    def _time_operation(self, operation: Callable, operation_name: str) -> Any:
        """Execute an operation with timing information."""
        start_time = time.time()
        try:
            result = operation()
            elapsed = time.time() - start_time
            logger.info("{} completed in {:.2f}s", operation_name, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("{} failed after {:.2f}s: {}", operation_name, elapsed, str(e))
            raise

    def open_ai_chat(self, prompt: str) -> Union[Tuple[str, Dict], str]:
        """Send a chat request to OpenAI API."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OPEN_AI.value)
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        def perform_openai_chat_operation():
            self._log_prompt(prompt, "OpenAI")

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str}
            ]

            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                logger.debug("Model {} doesn't support system messages, adapting prompt format", self.model)
                response_params = [
                    {"role": "user", "content": self.agent_prompt + prompt_str}
                ]

            model_id = self.model.split('/')[1]

            # Execute request with timing
            logger.debug("Executing chat completion with model: {}", model_id)
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=response_params,
                )
                elapsed = time.time() - start_time
                logger.info("OpenAI request completed in {:.2f}s", elapsed)

                # Log response information
                self._log_response_received(response, "OpenAI")

                # Check response validity
                check_response_result = self.check_response(response=response,
                                                            client_provider=ClientProvider.OPEN_AI.value)
                if check_response_result is not True:
                    return check_response_result

                # Process response
                if response.choices:
                    content = response.choices[0].message.content
                    content_length = len(content) if content else 0
                    logger.success("Response retrieved successfully (length: {})", content_length)
                    return content, response.usage
                else:
                    logger.warning("No choices in OpenAI response")
                    return "No response from the API."
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("OpenAI request failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_openai_chat_operation,
            client_provider=ClientProvider.OPEN_AI.value
        )

    def gpt_parse(self, prompt: str) -> Union[Tuple[Any, Dict], str, bool]:
        """Parse structured data using OpenAI's parsing capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OPEN_AI.value)
        if check_initial_result is not True:
            return check_initial_result

        if not self.parser:
            logger.error('Parser not configured for GPT parsing operation')
            return False

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        def perform_gpt_parse_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info("Parsing prompt with GPT: {}", truncated)
            logger.debug("Using parser: {}", self.parser.__name__)

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str}
            ]

            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                logger.debug("Model {} doesn't support system messages, adapting prompt format", self.model)
                response_params = [
                    {"role": "user", "content": self.agent_prompt + prompt_str},
                ]

            model_id = self.model.split('/')[1]

            # Execute request with timing
            logger.debug("Executing parse operation with model: {}", model_id)
            try:
                start_time = time.time()
                response = self.client.beta.chat.completions.parse(
                    model=model_id,
                    messages=response_params,
                    response_format=self.parser
                )
                elapsed = time.time() - start_time
                logger.info("GPT parse completed in {:.2f}s", elapsed)

                # Log response information
                logger.debug("Parse response received")
                logger.trace("Full parse response: {}", response)

                # Check response validity
                check_response_result = self.check_response(response=response,
                                                            client_provider=ClientProvider.OPEN_AI.value)
                if check_response_result is not True:
                    return check_response_result

                # Process response
                if response.choices:
                    parsed = response.choices[0].message.parsed
                    logger.success("Parsing successful with schema: {}", self.parser.__name__)
                    return parsed, response.usage
                else:
                    logger.warning("No parse choices in OpenAI response")
                    return "No response from the API."
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("GPT parse failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_gpt_parse_operation,
            client_provider=ClientProvider.OPEN_AI.value
        )

    """
        -----------------------------OPEN ROUTER
    """

    def open_router_chat(self, prompt: str) -> Union[Tuple[str, Dict], str]:
        """Send a chat request to OpenRouter API."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OPEN_ROUTER.value)
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_ROUTER)

        def perform_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "OpenRouter")

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt_str},
                    ],
                    temperature=0.1,
                )
                elapsed = time.time() - start_time
                logger.info("OpenRouter request completed in {:.2f}s", elapsed)

                # Log response information
                self._log_response_received(response, "OpenRouter")

                # Check response validity
                check_response_result = self.check_response(response=response,
                                                            client_provider=ClientProvider.OPEN_ROUTER.value)
                if check_response_result is not True:
                    return check_response_result

                # Process response
                content = response.choices[0].message.content
                usage_stats = getattr(response, 'usage', {})

                if not content:
                    logger.warning("Empty content in OpenRouter response")
                    return "Empty content in response."

                content_length = len(content) if content else 0
                logger.success("OpenRouter response retrieved successfully (length: {})", content_length)
                return content, usage_stats
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("OpenRouter request failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_chat_operation,
            client_provider=ClientProvider.OPEN_ROUTER.value
        )

    def open_parser(self, prompt: str) -> Union[Tuple[Any, Dict], str]:
        """Parse structured data using OpenRouter's parsing capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OPEN_ROUTER.value)
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_ROUTER)

        def perform_parse_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info("Parsing prompt with OpenRouter: {}", truncated)
            logger.debug("Using parser: {}", self.parser.__name__)

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.beta.chat.completions.parse(
                    model=f'{self.model}',
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt_str},
                    ],
                    response_format=self.parser.model_json_schema(),
                )
                elapsed = time.time() - start_time
                logger.info("OpenRouter parse completed in {:.2f}s", elapsed)

                # Log response information
                logger.debug("Parse response received from OpenRouter")
                logger.trace("Full parse response: {}", response)

                # Check response validity
                check_response_result = self.check_response(response=response,
                                                            client_provider=ClientProvider.OPEN_ROUTER.value)
                if check_response_result is not True:
                    return check_response_result

                # Process response
                parsed = response.choices[0].message.content
                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success("OpenRouter parsing successful with schema: {}", self.parser.__name__)
                    return result, response.usage
                except Exception as parse_error:
                    logger.error("OpenRouter parsing failed, falling back to default model: {}", parse_error)
                    logger.debug("Failed parse content: {}", self._truncate_text(str(parsed), 100))
                    # Fallback to default model
                    self.model = Models.O3_MINI.model_id
                    return self.gpt_parse(prompt=str(parsed))
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("OpenRouter parse failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_parse_operation,
            client_provider=ClientProvider.OPEN_ROUTER.value
        )

    """
        -----------------------------OLLAMA
    """

    def ollama_chat(self, prompt: str) -> Union[Tuple[str, Dict], str]:
        """Send a chat request to Ollama API."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OLLAMA.value)
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OLLAMA)

        def perform_ollama_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "Ollama")

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat(
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                )
                elapsed = time.time() - start_time
                logger.info("Ollama request completed in {:.2f}s", elapsed)

                # Log response information
                logger.info("Received response from Ollama")
                logger.trace("Full Ollama response: {}", response)

                # Process response
                if not response:
                    logger.warning("No response from Ollama API")
                    return "No response from the API."

                content = response.response
                usage = convert_ollama_to_openai_usage(ollama_output=response.dict())
                logger.debug("Ollama usage stats: {}", usage)

                content_length = len(content) if content else 0
                logger.success("Ollama response retrieved successfully (length: {})", content_length)
                return content, usage
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("Ollama request failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_ollama_chat_operation,
            client_provider=ClientProvider.OLLAMA.value
        )

    def ollama_parser(self, prompt: str) -> Union[Tuple[Any, Dict], str]:
        """Parse structured data using Ollama's capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(prompt=prompt, client_provider=ClientProvider.OLLAMA.value)
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OLLAMA)

        def perform_ollama_parser_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info("Parsing prompt with Ollama: {}", truncated)
            logger.debug("Using parser: {}", self.parser.__name__)

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat(
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    format=self.parser.model_json_schema(),
                )
                elapsed = time.time() - start_time
                logger.info("Ollama parse completed in {:.2f}s", elapsed)

                # Log response information
                logger.debug("Parse response received from Ollama")
                logger.trace("Full Ollama parse response: {}", response)

                # Process response
                if not response:
                    logger.warning("No parse response from Ollama API")
                    return "No response from the API."

                parsed = response.message.content
                usage = convert_ollama_to_openai_usage(ollama_output=response.dict())
                logger.debug("Ollama usage stats: {}", usage)

                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success("Ollama parsing successful with schema: {}", self.parser.__name__)
                    return result, usage
                except Exception as parse_error:
                    logger.error("Ollama parsing failed, falling back to default model: {}", parse_error)
                    logger.debug("Failed parse content: {}", self._truncate_text(str(parsed), 100))
                    # Fallback to default model
                    self.model = Models.O3_MINI.model_id
                    return self.gpt_parse(prompt=str(parsed))
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else -1
                logger.error("Ollama parse failed after {:.2f}s: {}", elapsed, str(e))
                raise

        return self.execute_with_error_handling(
            operation_func=perform_ollama_parser_operation,
            client_provider=ClientProvider.OLLAMA.value
        )

    """
        -----------------------------NON LLM
    """

    def whisper(self, file_path: str) -> Union[Tuple[str, Dict], str]:
        """
        Transcribe audio using OpenAI's Whisper model.

        Args:
            file_path (str): Path to the audio file to transcribe

        Returns:
            tuple[str, dict]: Tuple containing transcription text and usage stats
            str: Error message if transcription fails
        """
        logger.info("Transcribing audio file: {}", file_path)

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        try:
            # Verify file exists and is accessible
            if not os.path.exists(file_path):
                logger.error("Audio file not found: {}", file_path)
                return "Audio file not found."

            if not os.access(file_path, os.R_OK):
                logger.error("Audio file not readable: {}", file_path)
                return "Audio file not readable."

            # Get file size for logging
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.debug("Audio file size: {:.2f} MB", file_size)

            # Execute request with timing
            start_time = time.time()
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            elapsed = time.time() - start_time
            logger.info("Whisper transcription completed in {:.2f}s for {:.2f}MB file", elapsed, file_size)

            # Log response information
            logger.debug("Transcription response received")
            logger.trace("Full transcription response: {}", transcription)

            # Process response
            if hasattr(transcription, 'text'):
                text_length = len(transcription.text) if transcription.text else 0
                logger.success("Transcription completed successfully (length: {})", text_length)
                return transcription.text, getattr(transcription, 'usage', {})
            else:
                logger.warning("No transcription text in Whisper response")
                return "No transcription text in response."

        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else -1
            logger.error("Transcription failed after {:.2f}s: {}", elapsed, str(e))
            return f"An error occurred during transcription: {str(e)}"

    def embeddings(self, prompt: str) -> Union[Tuple[Any, Dict], str]:
        """Generate embeddings for the given prompt."""
        # Log prompt information
        truncated = self._truncate_text(prompt)
        logger.info("Generating embeddings: {}", truncated)

        # Set client
        self.set_client(ClientProvider.OPEN_AI)

        try:
            # Execute request with timing
            start_time = time.time()
            response = self.client.embeddings.create(
                model=self.model,
                input=prompt
            )
            elapsed = time.time() - start_time
            logger.info("Embeddings generation completed in {:.2f}s", elapsed)

            # Log response information
            logger.debug("Embeddings response received")
            logger.trace("Full embeddings response: {}", response)

            # Process response
            if response.data:
                embedding = response.data[0].embedding
                embedding_size = len(embedding)
                logger.success("Embeddings generated successfully (dimensions: {})", embedding_size)
                return embedding, response.usage
            else:
                logger.warning("No data in embeddings response")
                return "No response from the API."
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else -1
            logger.error("Embeddings generation failed after {:.2f}s: {}", elapsed, str(e))
            return f"An error occurred: {str(e)}"

    """
    -----------------------------UTILS
    """

    def set_client(self, client_provider: ClientProvider) -> None:
        """Set the appropriate client based on the provider."""
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

        logger.info("Setting client to {}", client_provider)

        client_factory = client_mapper.get(client_provider)
        if client_factory is None:
            error_msg = f"Unsupported client provider: {client_provider}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.client = client_factory()
        logger.success("Client successfully set to {}", client_provider)

    def check_initial(self, prompt: Any, client_provider: ClientProvider) -> Union[bool, str]:
        """Check initial conditions before sending request."""
        # Check prompt type
        if not isinstance(prompt, str):
            logger.error("Invalid prompt type: {}. Expected str.", type(prompt).__name__)
            return "Invalid prompt type. Expected string."

        # Check prompt content
        if not prompt.strip():
            logger.warning("Empty prompt received")
            return "Empty prompt received."

        # Check model
        if not self.model:
            logger.error("No model specified for {} request", client_provider)
            return "No model specified."

        logger.debug("Initial checks passed for {} request", client_provider)
        return True

    def check_response(self, response: Any, client_provider: ClientProvider) -> Union[bool, str]:
        """Check response validity."""
        # Check response existence
        if not response:
            logger.error("Received empty response from {}", client_provider)
            return "Empty response received from API."

        # Check response format
        if not hasattr(response, 'choices'):
            logger.error("Invalid response format from {}: missing 'choices' attribute", client_provider)
            return "Invalid response format from API."

        # Check choices content
        if not response.choices:
            logger.warning("No choices in response from {}", client_provider)
            return "No response choices available."

        logger.debug("Response checks passed for {} response", client_provider)
        return True

    def execute_with_error_handling(self, operation_func: Callable, client_provider: str) -> Any:
        """
        Execute an operation with standardized error handling.

        Args:
            operation_func: Callable function to execute within error handling
            client_provider: The client provider name for error messages

        Returns:
            The result of operation_func or an error message
        """
        try:
            return operation_func()
        except ConnectionError as e:
            logger.error("Connection error while calling {}: {}", client_provider, str(e))
            return f"Connection error: {str(e)}"
        except TimeoutError as e:
            logger.error("Timeout while calling {}: {}", client_provider, str(e))
            return f"Request timed out: {str(e)}"
        except Exception as e:
            logger.error("Error during {} operation: {}", client_provider, str(e))
            logger.exception("Full exception details")  # This logs the stack trace
            return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    run = AIDriver