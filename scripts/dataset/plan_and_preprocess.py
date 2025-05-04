from mgmGrowth.preprocessing.pipeline import ( # type: ignore
    planner,
    execute
) 

from typing import List

PLANNER_CONFIG: str = "/home/mariopasc/Python/Projects/Meningioma/src/mgmGrowth/preprocessing/configs/planner.yaml"
PATIENTS: str = "P1"
# P1, P12, P14, P15, P16, P19, P21, P22, P24, P26, P29, P35, P36, P40, P42, P45, P5, P50, P57, P6, P68, P7, P72, P73
OUTPUT_FOLDER: str = "/home/mariopasc/Python/Datasets/Meningiomas/processed/MenGrowth"
VERBOSE: bool = True
 
planner_json: str = planner.plan(yaml_path=PLANNER_CONFIG)
execute.preprocess(plan=planner_json,
                   patient_ids=PATIENTS,
                   output_dir=OUTPUT_FOLDER,
                   verbose=VERBOSE,)
