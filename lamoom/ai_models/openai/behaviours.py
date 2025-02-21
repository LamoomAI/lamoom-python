from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.ai_models.behaviour import AIModelsBehaviour
from lamoom.ai_models.openai.openai_models import (
    C_128K,
    C_4K,
    OpenAIModel,
    C_16K,
    C_32K,
)

OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-0125-preview",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        )
    
)

OPENAI_GPT4_1106_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-1106-preview",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        )
    
)

OPENAI_GPT4_1106_VISION_PREVIEW_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-1106-vision-preview",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        )
    
)

OPENAI_GPT4_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4",
                max_tokens=C_4K,
                support_functions=True,
            ),
            weight=100,
        )
)

OPENAI_GPT4_32K_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4-32k",
                max_tokens=C_32K,
                support_functions=True,
            ),
            weight=100,
        )
)

OPENAI_GPT3_5_TURBO_0125_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-3.5-turbo-0125",
                max_tokens=C_16K,
                support_functions=True,
            ),
            weight=100,
        )

)

OPENAI_GPT4O_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4o",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        )
)


OPENAI_GPT4O_MINI_BEHAVIOUR = AIModelsBehaviour(
    attempt=
        AttemptToCall(
            ai_model=OpenAIModel(
                model="gpt-4o-mini",
                max_tokens=C_128K,
                support_functions=True,
            ),
            weight=100,
        )
)
