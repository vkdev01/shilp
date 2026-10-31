"""Request and response bodies for both API surfaces.

They are separate models on purpose. Ollama's `/api/chat` and OpenAI's
`/v1/chat/completions` describe the same thing differently — nanoseconds and
`done: true` versus seconds and a `[DONE]` sentinel — and pretending otherwise
produces a schema that lies to one of the two clients. Translation happens in
the routes, once, visibly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    """Bodies tolerate unknown keys: clients send `keep_alive`, `format`, tools."""

    model_config = ConfigDict(extra="allow")


# --- Ollama: chat / generate ------------------------------------------
class OllamaMessage(_Open):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class OllamaChatRequest(_Open):
    model: str
    messages: list[OllamaMessage] = Field(min_length=1)
    stream: bool = True
    #: Sampling overrides; the manifest's params are the base layer.
    options: dict[str, Any] = Field(default_factory=dict)


class OllamaGenerateRequest(_Open):
    model: str
    prompt: str = ""
    system: str | None = None
    stream: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class OllamaChatResponse(_Open):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool = True
    done_reason: str | None = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaGenerateResponse(_Open):
    model: str
    created_at: str
    response: str = ""
    done: bool = True
    done_reason: str | None = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


# --- Ollama: model management -----------------------------------------
class OllamaModelDetails(_Open):
    format: str = "gguf"
    family: str = "unknown"
    families: list[str] = Field(default_factory=list)
    parameter_size: str = "unknown"
    quantization_level: str = "unknown"


class OllamaModel(_Open):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaTagsResponse(_Open):
    models: list[OllamaModel] = Field(default_factory=list)


class OllamaShowRequest(_Open):
    model: str


class OllamaShowResponse(_Open):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaModelDetails
    model_info: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class OllamaPullRequest(_Open):
    model: str
    stream: bool = True


class OllamaDeleteRequest(_Open):
    model: str


class OllamaEmbedRequest(_Open):
    model: str
    input: str | list[str]


class OllamaEmbedResponse(_Open):
    model: str
    embeddings: list[list[float]]


class OllamaPsResponse(_Open):
    models: list[dict[str, Any]] = Field(default_factory=list)


# --- OpenAI ------------------------------------------------------------
class OAIMessage(_Open):
    role: Literal["system", "user", "assistant", "tool", "developer"] = "user"
    content: str | None = ""


class OAIChatRequest(_Open):
    model: str
    messages: list[OAIMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    def options(self) -> dict[str, Any]:
        """Translate OpenAI names into engine parameters, dropping the unset ones."""
        stop = self.stop
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "seed": self.seed,
            "stop": [stop] if isinstance(stop, str) else stop,
        }


class OAIUsage(_Open):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIChoice(_Open):
    index: int = 0
    message: OAIMessage | None = None
    delta: OAIMessage | None = None
    finish_reason: str | None = None


class OAIChatResponse(_Open):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


class OAIModel(_Open):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "shilp"


class OAIModelList(_Open):
    object: str = "list"
    data: list[OAIModel] = Field(default_factory=list)


class OAICompletionRequest(_Open):
    model: str
    prompt: str
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None


class OAIEmbedRequest(_Open):
    model: str
    input: str | list[str]


class OAIEmbedResponse(_Open):
    object: str = "list"
    data: list[dict[str, Any]]
    model: str
    usage: OAIUsage
