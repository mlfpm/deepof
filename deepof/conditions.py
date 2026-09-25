"""Experimental conditions per video and, optionally, per animal.

Video-level conditions are stored as {experiment_id: 1-row DataFrame} with one column per condition.
Animal-level conditions are stored as {experiment_id: DataFrame indexed by animal id} with the same columns.
When animal-level conditions exist, the video-level conditions are derived from them: a video gets the
shared value if all its animals have the same one, and otherwise a composition of all values
(e.g. "control+stressed"), so all functions working on video-level conditions keep working.
"""

from typing import Dict, List, Optional

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
