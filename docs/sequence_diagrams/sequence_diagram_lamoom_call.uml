@startuml

    Note over Lamoom,LLM: Process of lamoom.call(prompt_id, context, model)

      Lamoom->>LibCache: get_prompt(prompt_id, version)
      alt Prompt is in cache or RECEIVE_PROMPT_FROM_SERVER disabled
          Lamoom-->>LibCache: Return cached prompt data
      end
      alt Cache miss or expired or new version of prompt is in the code
          Lamoom->>LamoomService: POST /lib/prompts (with currently active prompt data)

          Note right of LamoomService: Server checks if local prompt\nis the latest published version

          LamoomService-->>Lamoom: Response with prompt data and is_taken_globally flag

          Lamoom->>LibCache: Update cache with timestamp
          Note right of LibCache: Cache will be valid for 5 minutes\n(CACHE_PROMPT_FOR_EACH_SECONDS)
      end

      Note over Lamoom, LLM: Continue with AI model calling

      Lamoom-->>LLM: Call LLM w/ updated prompt and enriched context
      LLM ->> Lamoom: LLMResponse
@enduml
