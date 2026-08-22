from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pufferlab.contracts.retrieval import LexicalSpec, RetrievalConfig, RetrievalMode
from pydantic import ValidationError


def test_retrieval_mode_rejects_incompatible_specs() -> None:
    with pytest.raises(ValidationError, match="do not match mode"):
        RetrievalConfig(
            id=uuid4(),
            revision=1,
            name="bad vector config",
            dataset_version_id=uuid4(),
            mode=RetrievalMode.VECTOR,
            lexical=LexicalSpec(),
            config_hash="hash",
            created_at=datetime.now(UTC),
        )
