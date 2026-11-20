from pathlib import Path

from linked_ai_content_pipeline.config import load_config


def test_reference_config_exposes_grouped_scoring_rules() -> None:
    config = load_config(Path("configs/pipeline.toml"))

    assert [group.name for group in config.scoring.keyword_groups] == ["ai", "engineering"]
    assert config.scoring.combination_rules[0].required_groups == frozenset({"ai", "engineering"})
    assert config.scoring.noise_penalty == 0.15
    assert config.scoring.title_stopwords == ("weekly digest",)
    assert config.pipeline.candidate_pool_size == 10
    assert config.pipeline.diversity_penalty == 0.25
