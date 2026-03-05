"""Shared LLM client factory for all agent_tutorial steps."""
import os
import sys

from openai import OpenAI

KSYUN_BASE_URL = "https://kspmas.ksyun.com/v1/"
INFINI_BASE_URL = "https://cloud.infini-ai.com/maas/v1"


def create_client() -> tuple[OpenAI, str]:
    """Auto-detect provider from env vars. Returns (client, default_model).
    Ksyun takes priority when both keys are present."""
    if api_key := os.getenv("KSYUN_API_KEY"):
        base_url = os.getenv("KSYUN_BASE_URL", KSYUN_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url), "mco-4"
    if api_key := os.getenv("INFINI_API_KEY"):
        base_url = os.getenv("INFINI_BASE_URL", INFINI_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url), "deepseek-v3"
    print("Error: set KSYUN_API_KEY (Ksyun) or INFINI_API_KEY (InfiniAI).")
    sys.exit(1)


def list_models(client: OpenAI) -> None:
    """Fetch and display available models from the API."""
    print("Fetching available models...\n")
    try:
        models = client.models.list()
        model_list = sorted(models.data, key=lambda m: m.id)
        print(f"Found {len(model_list)} models:\n")
        for m in model_list:
            print(f"  - {m.id}")
        print()
    except Exception as e:
        print(f"Failed to list models: {e}")
        sys.exit(1)
