@startuml

  title Lamoom Test Creation Flow: Creating tests with ideal answers

  note over Lamoom,LamoomService: Process of creating tests for CI/CD validation

    alt Test creation via create_test method
      Lamoom->>LamoomService: create_test(prompt_id, context, ideal_answer)
      note right of LamoomService: Server creates test with:\n- Prompt ID\n- Test context\n- Ideal answer

      LamoomService-->>Lamoom: Return create test result
    end

    alt Test creation during normal prompt call with test_data
      Lamoom->>Lamoom: call(prompt_id, context, model, test_data={ideal_answer}),  Fetches prompt, calls LLM, gets response

      Lamoom->>SaveWorker: add_task(api_token, prompt_data, context, result, test_data)
      SaveWorker->>LamoomService: create_test_with_ideal_answer
      note right of LamoomService:  Server creates CI/CD test with:\n- Context\n- Prompt\n- Ideal answer
      Lamoom-->>Lamoom: Return AIResponse (without waiting for test creation)
    end

  @enduml