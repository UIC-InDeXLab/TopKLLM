import logging
from abc import ABC, abstractmethod
from docker import from_env
from docker.errors import DockerException
from docker.types import DeviceRequest

from models.singleton import Singleton

logger = logging.getLogger(__name__)

_OLLAMA_DOCKER_SUPPORTED_MODELS = [
    "phi3:medium-128k",
    "yi:9b-chat",
    "glm4:9b-chat-q4_0",
    "glm4:9b",
    "deepseek-coder-v2:16b-lite-instruct-q4_0",
    "qwen2:7b-instruct",
    "phi3:mini-4k",
    "yi:6b-chat",
    "qwen2:7b",
    "llama3:instruct",
    "gemma2:9b",
    "deepseek-coder-v2:lite",
    "zephyr:7b-beta",
    "qwen:14b-chat",
    "gemma:7b",
    "mistral:v0.1",
    "llama3:8b",
    "mistral:7b-instruct-v0.2-q4_0",
    "qwen:7b-chat",
    "mistral:v0.2",
    "qwen2:1.5b",
    "llama2:13b",
    "qwen2:1.5b-instruct",
    "llama2:7b",
    "qwen2:0.5b-instruct",
    "qwen2:0.5b",
    "gemma:2b"
]


class Response:
    def __init__(self, r):
        self._r = r

    @property
    def response(self):
        return self._r


class LLM(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def query(self, q, **kwargs) -> Response:
        pass


class OllamaDockerLLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = from_env()
        self.container = None

    @property
    def image_name(self):
        return "ollama/ollama"

    @property
    def name(self):
        return "ollama"

    def _start_container(self, envs=None, volumes=None):
        try:
            self.container = self.client.containers.run(
                image=self.image_name,
                detach=True,
                environment=envs,
                volumes=volumes,
                device_requests=[DeviceRequest(device_ids=["all"], capabilities=[['gpu']])]
            )
            logger.info("Container started")
        except DockerException as e:
            raise Exception(f"Failed to start container: {str(e)}")

    def _stop_container(self):
        if self.container:
            try:
                self.container.stop()
                logger.info("Container stopped")
            except DockerException as e:
                raise Exception(f"Failed to stop container: {str(e)}")

    def __enter__(self):
        self._start_container(envs={"./ollama": {'bind': '/root/.ollama', 'mode': 'rw'}})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_container()

    def query(self, q, **kwargs) -> Response:
        if not self.container:
            raise Exception("Ollama container is not running!")
        if kwargs.get("model", None) is None:
            raise Exception("You should specify llm model to run!")

        try:
            command = f'ollama run {kwargs.get("model").strip()} "{q}"'
            exit_code, output = self.container.exec_run(command, stderr=False)
            if exit_code != 0:
                raise Exception(f"Error in execution: {output.decode()}")
            return Response(output.decode())
        except DockerException as e:
            raise Exception(f"Failed to execute query: {str(e)}")
