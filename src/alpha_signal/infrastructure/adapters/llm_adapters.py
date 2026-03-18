"""llm_adapters.py — Adapters for LLM models using LangChain, Ollama, vLLM, and Llama.cpp."""

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alpha_signal.application.ports import SentimentPort, VlmPort
from alpha_signal.domain.models import SentimentResult, TradingSignal
from alpha_signal.infrastructure.logger import logger


# ============================================================================
# Helpers
# ============================================================================

def _encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_image_as_data_uri(image_path: Path) -> str:
    """Read an image and return a data URI for OpenAI-compatible APIs."""
    path = Path(image_path).resolve()
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(suffix, "image/png")
    b64 = _encode_image_to_base64(path)
    return f"data:{mime_type};base64,{b64}"


def _parse_json_response(raw_content: str, parser: PydanticOutputParser) -> Any:
    """Extract and parse JSON from LLM output that might contain markdown wrappers."""
    content = raw_content.strip()

    for attempt_fn in [
        lambda: parser.parse(content),
        lambda: parser.parse(content.split("```json")[1].split("```")[0].strip())
        if "```json" in content else None,
        lambda: parser.parse(content.split("```")[1].split("```")[0].strip())
        if "```" in content else None,
        lambda: parser.parse(content[content.find("{"):content.rfind("}") + 1])
        if "{" in content else None,
    ]:
        try:
            result = attempt_fn()
            if result is not None:
                return result
        except Exception:
            continue

    raise ValueError(
        f"Could not parse LLM response. "
        f"Raw response (first 500 chars): {content[:500]}"
    )


# ============================================================================
# Sentiment Adapter
# ============================================================================

class OllamaSentimentAdapter(SentimentPort):
    """Adapter for textual sentiment analysis using Ollama."""

    def __init__(self, base_url: str, model_name: str, temperature: float = 0.0):
        self._base_url = base_url
        self._model_name = model_name
        self._temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(f"Sentiment retry {rs.attempt_number}/3..."),
    )
    async def analyze_sentiment(self, news_text: str) -> SentimentResult:
        logger.info(f"Invoking Sentiment chain (Ollama {self._model_name})...")
        parser = PydanticOutputParser(pydantic_object=SentimentResult)
        
        llm = ChatOllama(
            base_url=self._base_url,
            model=self._model_name,
            temperature=self._temperature,
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Tu es un analyste de sentiment financier expert. "
                "Analyse le texte d'actualité et extrais le sentiment du marché. "
                "Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte "
                "avant ou après.\n\n"
                "{format_instructions}"
            ),
            (
                "human",
                "Analyse le sentiment de l'actualité financière suivante :\n\n"
                "{news_text}\n\n"
                "Réponds en JSON strict."
            ),
        ])

        chain = prompt | llm
        response = await chain.ainvoke({
            "format_instructions": parser.get_format_instructions(),
            "news_text": news_text,
        })

        sentiment = _parse_json_response(response.content, parser)
        logger.info(f"Sentiment: {sentiment.sentiment} (intensity: {sentiment.intensity:.2f})")
        return sentiment

    @property
    def provider_info(self) -> str:
        return f"ollama/{self._model_name}"


# ============================================================================
# VLM Adapters
# ============================================================================

class BaseVlmAdapter(VlmPort):
    """Base class for vision language models providing shared utilities."""
    
    def _build_system_prompt(self, format_instructions: str) -> str:
        return (
            "Tu es un analyste quantitatif senior dans un hedge fund de premier plan "
            "(type Jane Street). Tu analyses des graphiques financiers en chandeliers "
            "japonais (avec RSI et bandes de Bollinger) et le contexte d'actualité "
            "pour générer des signaux de trading.\n\n"
            "RÈGLES STRICTES:\n"
            "1. Réponds UNIQUEMENT avec un objet JSON valide.\n"
            "2. Pas de texte avant ou après le JSON.\n"
            "3. Utilise les indicateurs techniques visibles sur le graphique.\n"
            "4. Le stop_loss et take_profit doivent être réalistes.\n\n"
            f"{format_instructions}"
        )

    def _build_user_prompt(self, news_text: str) -> str:
        return (
            f"Analyse le graphique financier ci-dessus et le contexte "
            f"d'actualité suivant :\n\n"
            f"**Actualité** : {news_text}\n\n"
            f"Génère un signal de trading structuré au format JSON demandé."
        )


class LlamaCppVlmAdapter(BaseVlmAdapter):
    """Adapter for running fine-tuned VLM directly via llama.cpp (e.g., Qwen2-VL)."""

    def __init__(
        self, 
        model_path: str, 
        mmproj_path: str, 
        n_gpu_layers: int, 
        n_ctx: int, 
        temperature: float, 
        max_tokens: int
    ):
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = None  # Lazy load

    def _load_model(self):
        """Lazy load the llama_cpp model."""
        if self._model is None:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            
            logger.info("Loading Llama.cpp VLM model (this may take ~10s)...")
            chat_handler = Qwen25VLChatHandler(clip_model_path=self._mmproj_path)
            self._model = Llama(
                model_path=self._model_path,
                chat_handler=chat_handler,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                verbose=False,
            )
            logger.info("Llama.cpp VLM model loaded.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(f"VLM retry {rs.attempt_number}/3..."),
    )
    async def analyze_chart(self, image_path: Path, news_text: str) -> TradingSignal:
        logger.info(f"Invoking VLM via llama.cpp ({Path(self._model_path).name})...")
        self._load_model()
        
        parser = PydanticOutputParser(pydantic_object=TradingSignal)
        img_data_uri = _encode_image_as_data_uri(image_path)
        
        messages = [
            {"role": "system", "content": self._build_system_prompt(parser.get_format_instructions())},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_data_uri}},
                {"type": "text", "text": self._build_user_prompt(news_text)},
            ]}
        ]
        
        # llama_cpp python bindings are synchronous natively.
        # Run in executor to avoid blocking the async event loop
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _run_sync():
            return self._model.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            
        result = await loop.run_in_executor(None, _run_sync)
        raw_content = result["choices"][0]["message"]["content"]
        
        signal = _parse_json_response(raw_content, parser)
        logger.info(f"VLM Signal: {signal.action.value} (conf: {signal.confidence:.2f})")
        return signal

    @property
    def provider_info(self) -> str:
        return f"llama_cpp/{Path(self._model_path).name}"


class OllamaVlmAdapter(BaseVlmAdapter):
    """Adapter for running VLM (like llama3.2-vision) via Ollama."""

    def __init__(self, base_url: str, model_name: str, temperature: float = 0.0):
        self._base_url = base_url
        self._model_name = model_name
        self._temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(f"VLM retry {rs.attempt_number}/3..."),
    )
    async def analyze_chart(self, image_path: Path, news_text: str) -> TradingSignal:
        logger.info(f"Invoking VLM via Ollama ({self._model_name})...")
        parser = PydanticOutputParser(pydantic_object=TradingSignal)
        
        llm = ChatOllama(
            base_url=self._base_url,
            model=self._model_name,
            temperature=self._temperature,
        )

        img_data_uri = _encode_image_as_data_uri(image_path)
        
        messages = [
            SystemMessage(content=self._build_system_prompt(parser.get_format_instructions())),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": img_data_uri}},
                {"type": "text", "text": self._build_user_prompt(news_text)},
            ]),
        ]

        response = await llm.ainvoke(messages)
        signal = _parse_json_response(response.content, parser)
        logger.info(f"VLM Signal: {signal.action.value} (conf: {signal.confidence:.2f})")
        return signal

    @property
    def provider_info(self) -> str:
        return f"ollama/{self._model_name}"


class VllmVlmAdapter(BaseVlmAdapter):
    """Adapter for running fine-tuned VLM via vLLM."""

    def __init__(
        self, base_url: str, api_key: str, model_name: str, 
        temperature: float, max_tokens: int
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(f"VLM retry {rs.attempt_number}/3..."),
    )
    async def analyze_chart(self, image_path: Path, news_text: str) -> TradingSignal:
        from langchain_openai import ChatOpenAI
        logger.info(f"Invoking VLM via vLLM ({self._model_name})...")
        
        parser = PydanticOutputParser(pydantic_object=TradingSignal)
        img_data_uri = _encode_image_as_data_uri(image_path)
        
        llm = ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        messages = [
            SystemMessage(content=self._build_system_prompt(parser.get_format_instructions())),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": img_data_uri, "detail": "high"}},
                {"type": "text", "text": self._build_user_prompt(news_text)},
            ]),
        ]

        response = await llm.ainvoke(messages)
        signal = _parse_json_response(response.content, parser)
        logger.info(f"VLM Signal: {signal.action.value} (conf: {signal.confidence:.2f})")
        return signal

    @property
    def provider_info(self) -> str:
        return f"vllm/{self._model_name}"
