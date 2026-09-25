"""Experimental conditions per video and, optionally, per animal.

Video-level conditions are stored as {experiment_id: 1-row DataFrame} with one column per condition.
Animal-level conditions are stored as {experiment_id: DataFrame indexed by animal id} with the same columns.
When animal-level conditions exist, the video-level conditions are derived from them: a video gets the
shared value if all its animals have the same one, and otherwise a composition of all values
(e.g. "control+stressed"), so all functions working on video-level conditions keep working.

Supervised annotation columns name the animals they belong to ("B_climb-arena", "B_W_nose2tail"), so with
animal-level conditions each column can be attributed to a condition:
    - individual behaviors: the condition of that animal
    - directed pair behaviors (both "A_B_x" and "B_A_x" exist): the condition of the actor (first animal)
    - undirected pair behaviors (only "A_B_x" exists), depending on the pair policy:
        "composition" (default): the pair's composition, e.g. "control+stressed" (or "stressed" if both are)
        "both": counted once for each animal with its own condition (values are shared, not independent)
        "exclude_mixed": only pairs in which both animals share the condition
    - columns without animals: the video's composition
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ANIMAL_ID_COLUMN = "animal_id"
COMPOSITION_SEPARATOR = "+"


def find_animal_id_column(table: pd.DataFrame) -> Optional[str]:
    """Return the name of the animal id column of a condition table (case-insensitive), or None."""
    for col in table.columns[1:]:
        if str(col).strip().lower() == ANIMAL_ID_COLUMN:
            return col
    return None


def compose_condition(values) -> str:
    """Video-level value of one condition: the shared value, or all values sorted and joined."""
    values = sorted(str(v) for v in values)
    if len(set(values)) == 1:
        return values[0]
    return COMPOSITION_SEPARATOR.join(values)


def animal_conditions_from_table(table: pd.DataFrame, animal_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Build animal-level conditions from a long-format table.

    Args:
        table (pd.DataFrame): First column experiment ids, one animal id column, one column per condition.
        animal_ids (list): Animal ids of the project. If given, every experiment needs exactly these animals.

    Returns:
        dict: {experiment_id: DataFrame indexed by animal id, one column per condition}.
    """
    id_col = table.columns[0]
    animal_col = find_animal_id_column(table)
    if animal_col is None:
        raise ValueError(f'Animal-level conditions need an "{ANIMAL_ID_COLUMN}" column.')
    condition_cols = [c for c in table.columns if c not in (id_col, animal_col)]
    if not condition_cols:
        raise ValueError("No condition columns found next to the experiment and animal id columns.")

    animal_conditions = {}
    for exp_id, group in table.groupby(id_col, sort=False):
        df = group.set_index(animal_col)[condition_cols]
        df.index = df.index.astype(str)
        df.index.name = ANIMAL_ID_COLUMN
        animal_conditions[exp_id] = df

    validate_animal_conditions(animal_conditions, animal_ids)
    return animal_conditions


def validate_animal_conditions(animal_conditions: Dict[str, pd.DataFrame], animal_ids: Optional[List[str]] = None) -> None:
    """Check animal-level conditions for consistency; raises a ValueError listing all problems."""
    errors = []
    expected = None
    if animal_ids is not None:
        # Single-animal projects have the unnamed id "", so there is nothing to match against
        expected = {str(a) for a in animal_ids if str(a) != ""} or None

    for exp_id, df in animal_conditions.items():
        if df.index.duplicated().any():
            errors.append(f"{exp_id}: animal ids listed more than once: {sorted(set(df.index[df.index.duplicated()]))}")
        if expected is not None and set(df.index) != expected:
            errors.append(f"{exp_id}: animals {sorted(set(df.index))} do not match the project animals {sorted(expected)}")
        for col in df.columns:
            for animal, value in df[col].items():
                if not isinstance(value, str) or value.strip() == "":
                    errors.append(f"{exp_id}, {animal}, {col}: condition values need to be non-empty strings, got {value!r}")
                elif COMPOSITION_SEPARATOR in value:
                    errors.append(
                        f'{exp_id}, {animal}, {col}: condition values cannot contain "{COMPOSITION_SEPARATOR}" '
                        f"(reserved for videos with mixed conditions), got {value!r}"
                    )

    if errors:
        raise ValueError("Invalid animal-level conditions:\n  - " + "\n  - ".join(errors))


def derive_video_conditions(animal_conditions: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Derive video-level conditions ({experiment_id: 1-row DataFrame}) from animal-level conditions."""
    return {
        exp_id: pd.DataFrame([{col: compose_condition(df[col]) for col in df.columns}])
        for exp_id, df in animal_conditions.items()
    }


PAIR_POLICIES = ("composition", "both", "exclude_mixed")


def split_behavior_column(column: str, animal_ids: Sequence[str]) -> Tuple[Tuple[str, ...], str]:
    """Split a supervised annotation column into the animals involved and the behavior name.

    "B_climb-arena" -> (("B",), "climb-arena"), "B_W_nose2tail" -> (("B", "W"), "nose2tail"),
    columns without an animal prefix -> ((), column). Longer ids are matched first, so ids may contain "_".
    """
    ids = sorted({str(a) for a in animal_ids if str(a) != ""}, key=len, reverse=True)
    animals, rest = [], str(column)
    while len(animals) < 2:
        match = next((a for a in ids if rest.startswith(a + "_")), None)
        if match is None:
            break
        animals.append(match)
        rest = rest[len(match) + 1:]
    return tuple(animals), rest


def attach_animal_conditions(
    long_df: pd.DataFrame,
    animal_conditions: Dict[str, pd.DataFrame],
    exp_condition: str,
    animal_ids: Sequence[str],
    pair_policy: str = "composition",
    key_col: str = "exp_id",
    column_col: str = "cluster",
    condition_col: str = "exp condition",
    value_col: str = "time on cluster",
) -> pd.DataFrame:
    """Assign animal-level conditions to a long table of supervised annotation summaries.

    Args:
        long_df (pd.DataFrame): One row per experiment and annotation column, with key_col, column_col and value_col.
        animal_conditions (dict): {experiment_id: DataFrame indexed by animal id}.
        exp_condition (str): Condition to resolve.
        animal_ids (list): Animal ids of the project.
        pair_policy (str): Handling of undirected pair behaviors, see module docstring.

    Returns:
        pd.DataFrame: key_col, column_col (behavior name without animal prefix), condition_col and value_col,
        averaged per experiment, behavior and condition so that every experiment contributes at most one value per
        condition (animals of one video are not independent).
    """
    if pair_policy not in PAIR_POLICIES:
        raise ValueError(f"pair_policy needs to be one of {PAIR_POLICIES}, got {pair_policy!r}")

    columns = set(long_df[column_col].astype(str))
    rows = []
    for exp_id, column, value in zip(long_df[key_col], long_df[column_col].astype(str), long_df[value_col]):
        conds = animal_conditions[exp_id][exp_condition]
        animals, behavior = split_behavior_column(column, animal_ids)

        if len(animals) == 0:
            labels = [compose_condition(conds)]
        elif len(animals) == 1:
            labels = [conds[animals[0]]]
        elif f"{animals[1]}_{animals[0]}_{behavior}" in columns:  # directed: the actor comes first
            labels = [conds[animals[0]]]
        else:
            pair = [conds[animals[0]], conds[animals[1]]]
            if pair_policy == "composition":
                labels = [compose_condition(pair)]
            elif pair_policy == "both":
                labels = pair
            else:
                labels = [pair[0]] if pair[0] == pair[1] else []

        rows.extend((exp_id, behavior, str(label), value) for label in labels)

    out = pd.DataFrame(rows, columns=[key_col, column_col, condition_col, value_col])
    return out.groupby([key_col, column_col, condition_col], as_index=False, sort=False)[value_col].mean()


def comparison_design(df: pd.DataFrame, condition_a: str, condition_b: str, key_col: str = "exp_id", condition_col: str = "exp condition") -> str:
    """How two condition groups relate: "paired" (the same experiments contribute to both), "independent"
    (no shared experiments) or "mixed" (partly shared; neither paired nor independent tests apply)."""
    keys_a = set(df.loc[df[condition_col] == condition_a, key_col])
    keys_b = set(df.loc[df[condition_col] == condition_b, key_col])
    if keys_a == keys_b:
        return "paired"
    if not keys_a & keys_b:
        return "independent"
    return "mixed"
