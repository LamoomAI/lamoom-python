


from lamoom.prompt.chat import ChatsEntity


def test_chats_entity_resolve_not_multiple():
    ce = ChatsEntity(content="{greeting} World")
    context = {"greeting": "Hello"}
    resolved = ce.resolve(context)
    assert resolved[0].content == "Hello World"


def test_chats_entity_resolve_multiple():
    ce = ChatsEntity(content="{messages}", is_multiple=True)
    context = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "bot", "content": "Hello"},
        ]
    }
    resolved = ce.resolve(context)
    assert isinstance(resolved, list)
    assert len(resolved) == 2
    assert resolved[0].role == "user"
    assert resolved[0].content == "Hi"
    assert resolved[1].role == "bot"
    assert resolved[1].content == "Hello"


def test_chats_entity_resolve_not_exists():
    ce = ChatsEntity(content="{greeting} World")
    context = {"content": "Hello World"}
    resolved = ce.resolve(context)
    assert resolved[0].content == "{greeting} World"
    assert resolved[0].role == "user"


def test_chats_entity_get_values():
    ce = ChatsEntity(content="{greeting} World")
    context = {"greeting": "Hello"}
    values = ce.get_values(context)
    assert ce.add_in_reverse_order is False
    assert values[0].content == "Hello World"
    assert values[0].role == "user"
