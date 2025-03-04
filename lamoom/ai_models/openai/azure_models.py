import logging
import typing as t
from dataclasses import dataclass

from openai import AzureOpenAI

from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER
from lamoom.ai_models.openai.openai_models import FamilyModel, OpenAIModel

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AzureAIModel(OpenAIModel):
    realm: t.Optional[str]
    deployment_id: t.Optional[str]
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.AZURE
    model: t.Optional[str] = None

    def __str__(self) -> str:
        return f"{self.realm}-{self.deployment_id}-{self.family}"

    def __post_init__(self):
        if not self.family:
            if self.deployment_id.startswith("davinci"):
                self.family = FamilyModel.instruct_gpt.value
            elif self.deployment_id.startswith(("gpt3", "gpt-3")):
                self.family = FamilyModel.chat.value
            elif self.deployment_id.startswith("gpt-4o-mini"):
                self.family = FamilyModel.gpt4o_mini.value
            elif self.deployment_id.startswith("gpt-4o"):
                self.family = FamilyModel.gpt4o.value
            elif self.deployment_id.startswith(("gpt4", "gpt-4", "gpt")):
                self.family = FamilyModel.gpt4.value
            else:
                logger.info(
                    f"Unknown family for {self.deployment_id}. Please add it obviously. Setting as GPT4"
                )
                self.family = FamilyModel.gpt4.value
        logger.debug(f"Initialized AzureAIModel: {self}")

    @property
    def name(self) -> str:
        return f"{self.deployment_id}-{self.realm}"

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.deployment_id,
        }

    def get_client(self, client_secrets: dict = {}):
        print(f'Client secrets keys: {client_secrets.keys()}')
        realm_data = client_secrets.get(self.realm)
        if not realm_data:
            raise ValueError(f"Realm data for {self.realm} not found in client_secrets")
        return AzureOpenAI(
            api_version=realm_data.get("api_version", "2023-07-01-preview"),
            azure_endpoint=realm_data["azure_endpoint"],
            api_key=realm_data["api_key"],
        )

    def get_metrics_data(self):
        return {
            "realm": self.realm,
            "deployment_id": self.deployment_id,
            "family": self.family,
            "provider": self.provider.value,
        }
