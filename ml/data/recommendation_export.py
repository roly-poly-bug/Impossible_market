from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.data.recommendation_config import (
    DATASET_VERSION,
    DEFAULT_DATASET_SEED,
    ENGAGEMENT_VERSION,
    PRODUCT_VERSION,
    SESSION_EVENT_VERSION,
    SPLIT_WINDOWS,
    TASK_FAVORITEPLUS,
    TASK_PURCHASE,
    TASK_VIEWPLUS,
    TASKS,
    USER_VERSION,
)
from ml.data.recommendation_dataset import (
    InteractionFacts,
    RecommendationDatasetBundle,
    all_user_item_pairs,
    derive_task_state,
)


MASTER_COLUMNS = (
    "user_id",
    "product_id",
    "impression_count",
    "view_count",
    "favorite_count",
    "cart_count",
    "purchase_count",
    "was_impressed",
    "was_viewed",
    "was_favorited",
    "was_carted",
    "was_purchased",
    "first_impression_at",
    "first_view_at",
    "first_favorite_at",
    "first_cart_at",
    "first_purchase_at",
    "last_impression_at",
    "last_view_at",
    "last_favorite_at",
    "last_cart_at",
    "last_purchase_at",
    "first_interaction_at",
    "last_interaction_at",
)

TASK_COLUMNS = (
    "user_id",
    "product_id",
    "task",
    "state",
    "is_positive",
    "is_observed_non_conversion",
    "is_unknown",
    "impression_count",
    "view_count",
    "favorite_count",
    "cart_count",
    "purchase_count",
    "first_interaction_at",
    "last_interaction_at",
)


def _timestamp(value) -> str:
    return "" if value is None else value.isoformat().replace("+00:00", "Z")


def _facts_row(user_id: str, product_id: str, facts: InteractionFacts) -> dict[str, object]:
    return {
        "user_id": user_id,
        "product_id": product_id,
        "impression_count": facts.impression_count,
        "view_count": facts.view_count,
        "favorite_count": facts.favorite_count,
        "cart_count": facts.cart_count,
        "purchase_count": facts.purchase_count,
        "was_impressed": int(facts.was_impressed),
        "was_viewed": int(facts.was_viewed),
        "was_favorited": int(facts.was_favorited),
        "was_carted": int(facts.was_carted),
        "was_purchased": int(facts.was_purchased),
        "first_impression_at": _timestamp(facts.first_impression_at),
        "first_view_at": _timestamp(facts.first_view_at),
        "first_favorite_at": _timestamp(facts.first_favorite_at),
        "first_cart_at": _timestamp(facts.first_cart_at),
        "first_purchase_at": _timestamp(facts.first_purchase_at),
        "last_impression_at": _timestamp(facts.last_impression_at),
        "last_view_at": _timestamp(facts.last_view_at),
        "last_favorite_at": _timestamp(facts.last_favorite_at),
        "last_cart_at": _timestamp(facts.last_cart_at),
        "last_purchase_at": _timestamp(facts.last_purchase_at),
        "first_interaction_at": _timestamp(facts.first_interaction_at),
        "last_interaction_at": _timestamp(facts.last_interaction_at),
    }


def _write_master_csv(bundle: RecommendationDatasetBundle, path: Path) -> int:
    empty = InteractionFacts()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MASTER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        count = 0
        for user_id, product_id in all_user_item_pairs(
            bundle.world.user_ids, bundle.world.products
        ):
            writer.writerow(
                _facts_row(
                    user_id,
                    product_id,
                    bundle.full_facts.get((user_id, product_id), empty),
                )
            )
            count += 1
    return count


def _write_task_csv(
    bundle: RecommendationDatasetBundle,
    task: str,
    path: Path,
) -> int:
    train_facts = bundle.split_facts["train"]
    empty = InteractionFacts()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TASK_COLUMNS, lineterminator="\n")
        writer.writeheader()
        count = 0
        for user_id, product_id in all_user_item_pairs(
            bundle.world.user_ids, bundle.world.products
        ):
            facts = train_facts.get((user_id, product_id), empty)
            state = derive_task_state(task, facts)
            writer.writerow(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "task": task,
                    "state": state.state,
                    "is_positive": int(state.is_positive),
                    "is_observed_non_conversion": int(
                        state.is_observed_non_conversion
                    ),
                    "is_unknown": int(state.is_unknown),
                    "impression_count": facts.impression_count,
                    "view_count": facts.view_count,
                    "favorite_count": facts.favorite_count,
                    "cart_count": facts.cart_count,
                    "purchase_count": facts.purchase_count,
                    "first_interaction_at": _timestamp(facts.first_interaction_at),
                    "last_interaction_at": _timestamp(facts.last_interaction_at),
                }
            )
            count += 1
    return count


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_metadata(path: Path, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["data_rows"] = rows
    return result


def export_recommendation_dataset(
    bundle: RecommendationDatasetBundle,
    output_dir: str | Path,
    *,
    source_paths: dict[str, str | Path],
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}

    master_path = destination / "master_interactions.csv"
    master_rows = _write_master_csv(bundle, master_path)
    artifacts[master_path.name] = _artifact_metadata(master_path, rows=master_rows)

    for task in TASKS:
        task_path = destination / f"train_{task}.csv"
        task_rows = _write_task_csv(bundle, task, task_path)
        artifacts[task_path.name] = _artifact_metadata(task_path, rows=task_rows)

        seen_path = destination / f"train_seen_items_{task}.json"
        _write_json(
            seen_path,
            {
                "dataset_version": DATASET_VERSION,
                "task": task,
                "default_exclude_seen": True,
                "seen_definition": "train positive items for the same task",
                "items_by_user": bundle.train_seen_items[task],
            },
        )
        artifacts[seen_path.name] = _artifact_metadata(seen_path)

    for split in ("validation", "test"):
        for task in TASKS:
            relevance_path = destination / f"{split}_relevance_{task}.json"
            values = bundle.relevance[split][task]
            _write_json(
                relevance_path,
                {
                    "dataset_version": DATASET_VERSION,
                    "split": split,
                    "task": task,
                    "eligible_user_count": sum(bool(items) for items in values.values()),
                    "relevant_items_by_user": values,
                },
            )
            artifacts[relevance_path.name] = _artifact_metadata(relevance_path)

    for split, values in bundle.impressed_candidates.items():
        impressed_path = destination / f"impressed_candidates_{split}.json"
        _write_json(
            impressed_path,
            {
                "dataset_version": DATASET_VERSION,
                "split": split,
                "candidate_type": "observed_impression",
                "items_by_user": values,
            },
        )
        artifacts[impressed_path.name] = _artifact_metadata(impressed_path)

    candidate_path = destination / "candidate_sets.json"
    _write_json(
        candidate_path,
        {
            "dataset_version": DATASET_VERSION,
            "policies": bundle.all_item_candidates,
        },
    )
    artifacts[candidate_path.name] = _artifact_metadata(candidate_path)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": DEFAULT_DATASET_SEED,
        "source_versions": {
            "products": PRODUCT_VERSION,
            "users": USER_VERSION,
            "session_events": SESSION_EVENT_VERSION,
            "engagement": ENGAGEMENT_VERSION,
        },
        "splits": {
            window.name: {
                "start_inclusive": _timestamp(window.start),
                "end_exclusive": _timestamp(window.end_exclusive),
            }
            for window in SPLIT_WINDOWS
        },
        "tasks": {
            TASK_VIEWPLUS: "View events are positive; impressed without View is observed non-conversion.",
            TASK_FAVORITEPLUS: "Favorite, Cart, or Purchase is positive; View-only is observed non-conversion.",
            TASK_PURCHASE: "Purchase is positive; viewed without Purchase is observed non-conversion.",
        },
        "true_negative_policy": "No true-negative label is created. Unknown remains unknown.",
        "sampled_negative_policy": "Not implemented; sampled negative would not be a true negative.",
        "event_weights": None,
        "default_seen_policy": "Exclude same-task train-positive items; raw seen lists allow experiments to override this.",
        "candidate_policies": {
            task: {
                "policy": values["policy"],
                "excluded_statuses": values["excluded_statuses"],
                "item_count": len(values["product_ids"]),
            }
            for task, values in bundle.all_item_candidates.items()
        },
        "source_snapshots": {
            name: {
                "path": str(Path(path).as_posix()),
                "sha256": _sha256(Path(path)),
            }
            for name, path in sorted(source_paths.items())
        },
        "artifacts": dict(sorted(artifacts.items())),
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest
