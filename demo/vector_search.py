"""Run exact vector retrieval as one lazy, composable DuckPD plan."""

from __future__ import annotations

import duckpd as pd


def main() -> None:
    with pd.connect() as session:
        documents = session.sql(
            """
            SELECT * FROM (VALUES
                (1, 'earnings', [1.0, 0.0, 0.0]::FLOAT[3]),
                (2, 'earnings', [0.8, 0.2, 0.0]::FLOAT[3]),
                (3, 'macro', [0.0, 1.0, 0.0]::FLOAT[3]),
                (4, 'macro', [-1.0, 0.0, 0.0]::FLOAT[3])
            ) documents(document_id, topic, embedding)
            """
        )
        eligible = documents[documents["topic"] == "earnings"]
        matches = eligible.vector.search(
            [1.0, 0.0, 0.0],
            column="embedding",
            metric="cosine",
            k=2,
            tie_breaker="document_id",
        )

        print(f"Executions after planning: {session.execution_count}")
        print(matches.explain("logical"))
        print(matches.collect().to_string(index=False))
        print(f"Executions after collection: {session.execution_count}")


if __name__ == "__main__":
    main()
