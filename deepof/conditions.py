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


def column_condition_labels(
    column: str,
    all_columns,
    conditions: pd.Series,
    animal_ids: Sequence[str],
    pair_policy: str = "composition",
) -> Tuple[str, List[str]]:
    """Behavior name and condition label(s) of one supervised annotation column in one video.

    Args:
        column (str): Annotation column, e.g. "B_W_nose2nose".
        all_columns: All annotation columns (a pair behavior is directed if the reversed column exists too).
        conditions (pd.Series): Condition value per animal id for this video.
        animal_ids (list): Animal ids of the project.
        pair_policy (str): Handling of undirected pair behaviors, see module docstring.

    Returns:
        (behavior name without animal prefixes, list of condition labels; empty if excluded)
    """
    if pair_policy not in PAIR_POLICIES:
        raise ValueError(f"pair_policy needs to be one of {PAIR_POLICIES}, got {pair_policy!r}")
    animals, behavior = split_behavior_column(column, animal_ids)

    if len(animals) == 0:
        labels = [compose_condition(conditions)]
    elif len(animals) == 1:
        labels = [conditions[animals[0]]]
    elif f"{animals[1]}_{animals[0]}_{behavior}" in all_columns:  # directed: the actor comes first
        labels = [conditions[animals[0]]]
    else:
        pair = [conditions[animals[0]], conditions[animals[1]]]
        if pair_policy == "composition":
            labels = [compose_condition(pair)]
        elif pair_policy == "both":
            labels = pair
        else:
            labels = [pair[0]] if pair[0] == pair[1] else []

    return behavior, [str(label) for label in labels]


def animals_with_condition(animal_conditions: Dict[str, pd.DataFrame], exp_condition: str, value: str) -> Dict[str, List[str]]:
    """{experiment_id: animal ids with the given condition value}, only experiments with at least one such animal."""
    matching = {
        exp_id: [str(a) for a, v in df[exp_condition].items() if str(v) == str(value)]
        for exp_id, df in animal_conditions.items()
    }
    return {exp_id: animals for exp_id, animals in matching.items() if animals}


def select_animal_rows(table: pd.DataFrame, names: Sequence[str], animals: Sequence[str], animal_ids: Sequence[str]) -> pd.DataFrame:
    """Stack the data of selected animals below each other, one block of rows per animal.

    Names without an animal prefix ("Center") are pooled over the selected animals (column "Center" holds "A_Center"
    in A's rows, "B_Center" in B's rows); prefixed names ("A_Center") only hold data in the rows of that animal and
    NaN elsewhere. Works for any table whose (first level) columns are "<animal>_<name>".
    """
    blocks = []
    for animal in animals:
        pieces = {}
        for name in names:
            prefix, base = split_behavior_column(name, animal_ids)
            source = f"{animal}_{base}" if not prefix else name
            piece = table[source].copy()
            if prefix and prefix[0] != animal:
                piece.loc[:] = float("nan")
            pieces[name] = piece
        blocks.append(pd.concat(pieces, axis=1))
    return pd.concat(blocks, axis=0, ignore_index=True)


def expand_behaviors(behaviors: Sequence[str], columns: Sequence[str], animal_ids: Sequence[str]) -> List[str]:
    """Annotation columns selected by behavior names, which can be column names ("B_climb-arena") or
    behavior names pooled over animals ("climb-arena")."""
    wanted = set(behaviors)
    return [
        col for col in columns
        if col in wanted or split_behavior_column(col, animal_ids)[1] in wanted
    ]


def pooled_behavior_names(columns: Sequence[str], animal_ids: Sequence[str]) -> List[str]:
    """Behavior names of annotation columns with the animal prefixes removed, in order of appearance."""
    return list(dict.fromkeys(split_behavior_column(col, animal_ids)[1] for col in columns))


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
    columns = set(long_df[column_col].astype(str))
    rows = []
    for exp_id, column, value in zip(long_df[key_col], long_df[column_col].astype(str), long_df[value_col]):
        behavior, labels = column_condition_labels(
            column, columns, animal_conditions[exp_id][exp_condition], animal_ids, pair_policy
        )
        rows.extend((exp_id, behavior, label, value) for label in labels)

    out = pd.DataFrame(rows, columns=[key_col, column_col, condition_col, value_col])
    return out.groupby([key_col, column_col, condition_col], as_index=False, sort=False)[value_col].mean()


def summary_to_animal_rows(
    summary: pd.DataFrame,
    animal_conditions: Dict[str, pd.DataFrame],
    exp_condition: str,
    animal_ids: Sequence[str],
    key_col: str = "experiment_id",
    video_col: str = "video_id",
    meta_cols: Sequence[str] = ("bin_number",),
) -> pd.DataFrame:
    """Turn a supervised summary with one row per video (and time bin) into one row per animal.

    Each animal gets its individual behaviors, the directed pair behaviors in which it is the actor and the undirected
    pair behaviors it takes part in (averaged over partners), all under pooled names ("climb-arena", "following"), and
    its own condition value. key_col becomes "<video>_<animal>" (the unit of analysis), video_col keeps the video.
    Columns that are neither meta columns, key_col nor behaviors (e.g. other conditions) are dropped.
    """
    condition_names = set(next(iter(animal_conditions.values())).columns)
    behavior_cols = [
        c for c in summary.columns
        if c not in set(meta_cols) | {key_col, video_col} | condition_names
    ]
    all_columns = set(behavior_cols)
    named = [str(a) for a in animal_ids if str(a) != ""]

    # columns contributing to each pooled behavior, per animal
    sources = {a: {} for a in named}
    for col in behavior_cols:
        animals, behavior = split_behavior_column(col, animal_ids)
        if len(animals) == 0:
            owners = named  # video-level columns belong to every animal
        elif len(animals) == 1:
            owners = [animals[0]]
        elif f"{animals[1]}_{animals[0]}_{behavior}" in all_columns:  # directed: only the actor
            owners = [animals[0]]
        else:  # undirected: both participants
            owners = list(animals)
        for a in owners:
            sources[a].setdefault(behavior, []).append(col)

    kept_meta = [c for c in meta_cols if c in summary.columns]
    blocks = []
    for a in named:
        block = summary[kept_meta].copy()
        block[video_col] = summary[key_col].values
        block[key_col] = [f"{v}_{a}" for v in summary[key_col]]
        block[exp_condition] = [str(animal_conditions[v].loc[a, exp_condition]) for v in summary[key_col]]
        for behavior, cols in sources[a].items():
            block[behavior] = summary[cols].mean(axis=1).values
        blocks.append(block)
    out = pd.concat(blocks, axis=0, ignore_index=True)
    return out[kept_meta + [key_col, video_col, exp_condition] + [c for c in out.columns if c not in set(kept_meta) | {key_col, video_col, exp_condition}]]


def condition_design(df: pd.DataFrame, conditions: Sequence[str], key_col: str = "exp_id", condition_col: str = "exp condition") -> str:
    """How several condition groups relate: "paired" (every experiment contributes to every condition),
    "independent" (no experiment contributes to more than one) or "mixed" (neither)."""
    key_sets = [set(df.loc[df[condition_col].astype(str) == str(c), key_col]) for c in conditions]
    if all(ks == key_sets[0] for ks in key_sets):
        return "paired"
    if sum(len(ks) for ks in key_sets) == len(set().union(*key_sets)):
        return "independent"
    return "mixed"


def video_conditions(
    exp_conditions: Dict[str, pd.DataFrame],
    exp_condition: Optional[str] = None,
    animal_conditions: Optional[Dict[str, pd.DataFrame]] = None,
    condition_source: str = "composition",
    keys=None,
) -> Dict[str, str]:
    """One condition value per video, for analyses of data that belongs to whole videos (e.g. soft counts).

    Args:
        exp_conditions (dict): Video-level conditions ({experiment_id: 1-row DataFrame}).
        exp_condition (str): Condition to use. Defaults to the first one.
        animal_conditions (dict): Animal-level conditions, needed for condition_source other than "composition".
        condition_source (str): "composition" (default): the video-level value, i.e. the shared value or the
            composition of mixed videos (e.g. "control+stressed"); "exclude_mixed": like "composition", but videos
            whose animals have different values are left out; an animal id (e.g. "B"): that animal's value.
        keys: Videos to include (default: all).

    Returns:
        dict: {experiment_id: condition value}; videos left out by condition_source are missing.
    """
    if exp_condition is None:
        exp_condition = next(iter(exp_conditions.values())).columns[0]
    keys = list(exp_conditions.keys()) if keys is None else list(keys)

    if condition_source == "composition" or (condition_source == "exclude_mixed" and animal_conditions is None):
        return {k: str(exp_conditions[k][exp_condition].values[0]) for k in keys if k in exp_conditions}

    if animal_conditions is None:
        raise ValueError(
            f'condition_source="{condition_source}" needs animal-level conditions (load them with an "{ANIMAL_ID_COLUMN}" column).'
        )
    if condition_source == "exclude_mixed":
        uniform = {
            k: str(animal_conditions[k][exp_condition].iloc[0])
            for k in keys
            if k in animal_conditions and animal_conditions[k][exp_condition].astype(str).nunique() == 1
        }
        if keys and not uniform:
            raise ValueError(
                f'condition_source="exclude_mixed" leaves no videos: all animals within each video differ in "{exp_condition}".'
            )
        return uniform

    animal = str(condition_source)
    missing = [k for k in keys if k in animal_conditions and animal not in animal_conditions[k].index]
    if missing:
        raise ValueError(
            f'condition_source="{animal}" needs to be "composition", "exclude_mixed" or an animal id; '
            f"{animal!r} is not an animal of {missing[:3]}."
        )
    return {k: str(animal_conditions[k].loc[animal, exp_condition]) for k in keys if k in animal_conditions}


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
