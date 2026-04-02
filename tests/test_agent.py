from agt.agent import Agent, FakeLLMClient, Tool, echo_tool


def test_agent_returns_fake_llm_message() -> None:
    agent = Agent(llm=FakeLLMClient())
    agent.register_tool(Tool("echo", "Echo input text", echo_tool))

    result = agent.run("hello")

    assert "[fake-llm]" in result
    assert "hello" in result
    assert agent.messages[0].role == "user"
    assert agent.messages[-1].role == "assistant"


def test_load_skills_finds_skill_md(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "writing"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# writing", encoding="utf-8")

    agent = Agent()
    skills = agent.load_skills(tmp_path / "skills")

    assert skills == ["writing"]


def test_fake_llm_streams_multiple_chunks() -> None:
    llm = FakeLLMClient()
    from agt.agent import Message

    chunks = list(llm.stream_text([Message(role="user", content="x")]))

    assert len(chunks) >= 2
    assert "".join(chunks).startswith("[fake-llm]")
