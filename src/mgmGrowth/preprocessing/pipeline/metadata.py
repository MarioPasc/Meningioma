import json
import pandas as pd # type: ignore
from typing import Any, Dict
import os
import re

HARDCODED_COLUMNS_DICT: Dict[str, str] = {
    "Unnamed: 0": "general",
    "0 nada; 1 cirugía; 2 atrofia; 3: infarto; 4 hemorragia; 5 aneurisma; 6 cáncer; 7 trauma": "general",
    "Unnamed: 2": "general",
    "0: hombre 1:mujer": "general",
    "Unnamed: 4": "first_study/rm",
    "Unnamed: 5": "first_study/rm",
    "Unnamed: 6": "first_study/tc",
    "Unnamed: 7": "first_study/tc",
    "MEDIDAS REFERENCIA  PRIMER ESTUDIO ": "first_study/measurements",
    "Unnamed: 9": "first_study/measurements",
    "Unnamed: 10": "first_study/measurements",
    "VOLUMEN ": "first_study/measurements",
    "Unnamed: 12": "first_study/attributes",
    "PRIMER  ESTUDIO": "first_study/attributes",
    "Unnamed: 14": "first_study/measurements",
    "0: NO 1:<25% 2:25-50% 3:>50% 4:completo": "first_study/attributes",
    "Unnamed: 16": "first_study/attributes",
    "Unnamed: 17": "first_study/attributes",
    "0: fosa posterior.\n1: ala esfenoidal.\n2: hoz cerebral.\n3: plano etmoidal / oflatorio 4 convexidad 5 seno cavern": "first_study/attributes",
    "0 no, 1 periférica, 2 central, 3 mixta, 4 total": "first_study/measurements",
    "1ER CONTROL": "c1",
    "Unnamed: 21": "c1",
    "Unnamed: 22": "c1",
    "Unnamed: 23": "c1",
    "Unnamed: 24": "c1",
    "Unnamed: 25": "c1",
    "Unnamed: 26": "c1",
    "2DO": "c2",
    "Unnamed: 28": "c2",
    "Unnamed: 29": "c2",
    "Unnamed: 30": "c2",
    "Unnamed: 31": "c2",
    "Unnamed: 32": "c2",
    "Unnamed: 33": "c2",
    "TERCERO": "c3",
    "Unnamed: 35": "c3",
    "Unnamed: 36": "c3",
    "Unnamed: 37": "c3",
    "Unnamed: 38": "c3",
    "Unnamed: 39": "c3",
    "Unnamed: 40": "c3",
    "Unnamed: 41": "c3",
    "CUARTO": "c4",
    "Unnamed: 43": "c4",
    "Unnamed: 44": "c4",
    "Unnamed: 45": "c4",
    "Unnamed: 46": "c4",
    "Unnamed: 47": "c4",
    "Unnamed: 48": "c4",
    "QUINTO": "c5",
    "Unnamed: 50": "c5",
    "Unnamed: 51": "c5",
    "Unnamed: 52": "c5",
    "Unnamed: 53": "c5",
    "Unnamed: 54": "c5",
    "Unnamed: 55": "c5",
    "PROGR CALCIF": "groundtruth",
    "CRECE": "groundtruth",
}

HARCODED_SUBCOLUMNS_DIC: Dict[str, str] = {
    "paciente": "ID",
    "antec": "medical_history",
    "edad": "age",
    "sexo": "sex",
    "fecha rm": "date",
    "equipo": "machine",
    "fecha tc": "date",
    "tec tc": "tec",
    "cc": "cc",
    "ll": "ll",
    "ap": "ap",
    "volumen": "vol",
    "lobulado si(1)/no(0)": "lobed",
    "hiperostosis si(1)/no(0)": "hiperostosis",
    "edema si(1)/no (0)": "edema",
    "escala visual de calcificacion": "visual_calcification_scale",
    "señal t2 (0 isointenso, 1 heterogeneo, 2 hipo, 3 hiper)": "t2_signal",
    "patron realce (0 homogeneo fuerte, 1 heterogéneo, 2 hipocaptante)": "enhancement_pattern",
    "localizacion": "loc",
    "tipo calcif": "calcif",
    "fecha": "date",
    "vol": "vol",
    "calcificacion": "calcif",
    "edema": "edema",
    "progr calcif": "progr_calcif",
    "crece": "growth",
}


def normalize_text(text: Any) -> Any:
    """
    Cleans text values:
    - Strips leading/trailing spaces
    - Converts to lowercase
    - Replaces multiple spaces with a single space
    - Removes special hidden characters (non-breaking spaces)
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\xa0", " ")  # Handle non-breaking spaces
        return text.lower()
    return text


def apply_hardcoded_codification(xlsx_path: str, output_csv_path: str) -> None:
    """
    Apply the hardcoded codification to the first-grade and second-grade columns of the
    xlsx file, in order to generate a first-stage clean csv file.
    """
    df = pd.read_excel(xlsx_path)
    # Rename top-level columns
    df.rename(columns=HARDCODED_COLUMNS_DICT, inplace=True)

    # Prepare normalized subcolumn keys
    hardcoded_subcolumns_clean = {
        normalize_text(k): v for k, v in HARCODED_SUBCOLUMNS_DIC.items()
    }

    # Normalize the row (index=0) that contains subcolumn names
    df.iloc[0] = df.iloc[0].apply(normalize_text)

    # Replace them using the dictionary
    df.iloc[0] = df.iloc[0].replace(hardcoded_subcolumns_clean, regex=False)

    df.to_csv(path_or_buf=output_csv_path, index=False)


def is_zero_or_none(value: Any) -> bool:
    """
    Returns True if 'value' is None, or zero, or the string "0", else False.
    """
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip() == "0"
    return False


def is_control_block_empty(c_dict: Dict[str, Any]) -> bool:
    """
    Returns True if:
      - The "vol" field is None or zero (int/float) or the string "0"
      AND
      - All other fields are None
    Otherwise returns False.
    """
    vol_val = c_dict.get("vol", None)

    # If vol is not zero/None, the block is definitely not empty
    if not is_zero_or_none(vol_val):
        return False

    # Now check that every other field (besides "vol") is None
    for k, v in c_dict.items():
        if k == "vol":
            continue
        if v is not None:
            return False

    # If we reached here, vol was zero/None, and every other field is None
    return True


def create_json_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads a CSV that has:
        - Row 0: top-level grouping (e.g., 'general', 'first_study/rm', 'c1', 'groundtruth', etc.)
        - Row 1: second-level column names (keys under those groupings)
        - Rows 2+: patient data rows

    The script constructs a JSON of the form:
    {
        "P{patient_id}": {
            "general": { ... },
            "first_study": {
                "rm": { ... },
                "tc": { ... },
                "attributes": { ... },
                "measurements": { ... }
            },
            "c1": { ... },
            "c2": { ... },
            ...
            "groundtruth": { ... }
        },
        ...
    }

    The patient_id is taken from the column where top_level == 'general'
    and second_level == 'ID'. Missing/NaN values become null in the JSON.

    :param csv_path: Path to the input CSV file.
    :return: A dictionary that can be serialized to JSON.
    """
    # Read the entire CSV without interpreting headers
    df = pd.read_csv(csv_path, header=None)

    # Row 0 -> top-level group names
    top_level_cols = df.iloc[0].tolist()
    # Row 1 -> second-level column names
    second_level_cols = df.iloc[1].tolist()

    # The data for patients starts from row 2 onward
    data = df.iloc[2:].reset_index(drop=True)

    # Identify 'general/ID' column index
    id_column_index = None
    for i, (top_val, second_val) in enumerate(zip(top_level_cols, second_level_cols)):
        if top_val == "general" and second_val == "ID":
            id_column_index = i
            break

    if id_column_index is None:
        raise ValueError(
            "Could not find the 'general/ID' column in the CSV headers. "
            "Ensure the CSV has a top-level 'general' with second-level 'ID'."
        )

    # Prepare a structure to map columns into the JSON hierarchy
    column_map = []
    for top_val, second_val in zip(top_level_cols, second_level_cols):
        if "/" in str(top_val):
            parts = top_val.split("/", 1)
            top_key = parts[0]
            sub_key = parts[1]
        else:
            top_key = top_val
            sub_key = None

        column_map.append((top_key, sub_key, second_val))

    result: Dict[str, Dict[str, Any]] = {}

    # Process each patient row
    for row_idx in range(data.shape[0]):
        row_data = data.iloc[row_idx]

        # Retrieve patient ID from 'general/ID'
        raw_pid = row_data[id_column_index]
        if pd.isnull(raw_pid):
            # If no ID, skip
            continue

        # Convert ID to e.g., "P1"
        patient_id_str = f"P{int(raw_pid)}"
        patient_dict: Dict[str, Any] = {}

        # Fill data
        for col_idx, (top_key, sub_key, second_key) in enumerate(column_map):
            if not top_key:
                # Skip if top_key is empty
                continue

            val = row_data[col_idx]
            if pd.isnull(val):
                val = None

            if sub_key is None:
                # E.g. "general" -> "ID": val
                if top_key not in patient_dict:
                    patient_dict[top_key] = {}
                patient_dict[top_key][second_key] = val
            else:
                # E.g. "first_study" -> "rm" -> "date": val
                if top_key not in patient_dict:
                    patient_dict[top_key] = {}
                if sub_key not in patient_dict[top_key]:
                    patient_dict[top_key][sub_key] = {}
                patient_dict[top_key][sub_key][second_key] = val

        # Check c1..c5 blocks; remove if empty
        for cx in ["c1", "c2", "c3", "c4", "c5"]:
            if cx in patient_dict and isinstance(patient_dict[cx], dict):
                if is_control_block_empty(patient_dict[cx]):
                    del patient_dict[cx]

        result[patient_id_str] = patient_dict

    return result


def main() -> None:
    """
    Main function to run when calling this script from the command line.
    It reads an XLSX file, applies codification, writes an intermediate CSV,
    then reads that CSV to build a final JSON structure.
    """
    xlsx_path = "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/raw/processed/baseline/metadata.xlsx"
    output_folder = (
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/"
    )
    csv_path = os.path.join(output_folder, "metadata_recodified.csv")

    # 1. Apply the hardcoded codification
    apply_hardcoded_codification(
        xlsx_path=xlsx_path,
        output_csv_path=csv_path,
    )

    # 2. Create JSON from the CSV
    json_data = create_json_from_csv(csv_path)

    # 3. Write the JSON structure to file
    output_json = os.path.join(output_folder, "metadata_clean.json")
    with open(output_json, "w") as outfile:
        json.dump(json_data, outfile, indent=2)


if __name__ == "__main__":
    main()
