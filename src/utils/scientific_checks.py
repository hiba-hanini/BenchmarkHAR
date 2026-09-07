"""Small, explicit invariants for subject-aware benchmark evaluation."""


def class_counts_for_labels(labels, class_ids) -> dict:
    class_ids = [int(class_id) for class_id in class_ids]
    counts = {int(class_id): 0 for class_id in class_ids}
    for value in labels:
        value = int(value)
        if value in counts:
            counts[value] += 1
    return counts


def validate_subject_split(train_subject_ids, test_subject_ids) -> None:
    train = set(train_subject_ids)
    test = set(test_subject_ids)
    if not train or not test:
        raise ValueError("Each fold must contain at least one train and one test subject.")
    overlap = train & test
    if overlap:
        raise AssertionError(f"Subject leakage detected: {sorted(overlap)}")


def validate_fold_samples(X_train, X_test) -> None:
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Each fold must contain non-zero train and test samples.")


def validate_label_mapping(labels, class_ids, split_name: str) -> None:
    unknown = set(labels) - set(class_ids)
    if unknown:
        raise AssertionError(f"{split_name} labels outside canonical vocabulary: {sorted(unknown)}")
