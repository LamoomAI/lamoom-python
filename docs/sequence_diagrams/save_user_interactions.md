sequenceDiagram
    title Lamoom Logging Flow: Recording user interactions with prompts
    participant Lamoom
    participant SaveWorker
    participant LamoomService

    Note over Lamoom, LamoomService: Process of recording prompt execution logs
    Lamoom->>Lamoom: call(prompt_id, context, model) - Fetches prompt, calls LLM, gets response
    Note right of SaveWorker: Async worker for saving interactions to avoid blocking
    Lamoom->>SaveWorker: add_task(api_token, prompt_data, context, result, test_data)
    SaveWorker->>LamoomService: save_user_interaction(api_token, prompt_data, context, response)
    Note right of LamoomService: POST /lib/logs: Records interaction for:\n- Analytics\n- Debugging\n- Performance tracking\n- Cost monitoring
    Lamoom-->>Lamoom: Return AIResponse (without waiting for log completion)
