# Copyright 2020 DeepCode AG

# Author: Berkay Berabi
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Instruction:
    type: int
    text: str
    line_number: int
    line_column: int
    global_idx: int
    description: str


@dataclass
class RuleReport:
    rule_id: str
    message: str
    col_begin: int
    col_end: int
    line_begin: int
    line_end: int
    severity: int


@dataclass
class DataPointT5Representation:
    """A representation for the datapoint, suitable for training T5 model.

    Included to avoid tuples, and for readability.
    """

    input: str
    output: str


@dataclass
class DataPoint:
    source_code: str
    target_code: str
    warning_line: str
    rule_report: RuleReport
    instructions: List[Instruction]
    initial_source_patch: str
    initial_target_patch: str
    repo: str
    source_filename: str
    target_filename: str
    source_changeid: str
    target_changeid: str

    # Optional fields which are only used in test scripts.
    predictions: Optional[List] = None
    first_fixed: Optional[int] = None
    beam_fixed: Optional[int] = None
    fix_type: Optional[str] = None

    def serialize_instructions(self) -> str:
        serialize_instructions = ""
        for inst in self.instructions:
            if inst.type == 0:
                type_name = "delete"
            elif inst.type == 1:
                type_name = "insert"
            else:
                raise AttributeError("Unknown instruction type")

            serialize_instructions += type_name + " " + inst.text + "\n"
        return serialize_instructions

    def get_description(self) -> str:
        desc = (
            "WARNING\n"
            + self.rule_report.rule_id
            + " "
            + self.rule_report.message
            + " at line: "
            + str(self.rule_report.line_begin)
            + "\n"
        )

        desc += "WARNING LINE\n" + self.warning_line + "\n"
        desc += (
            "SOURCE PATCH\n"
            + self.source_code
            + "\nTARGET PATCH\n"
            + self.target_code
            + "\n"
        )

        desc += "INSTRUCTIONS\n"
        for inst in self.instructions:
            desc += inst.description + "\n"
        return desc

    def get_t5_representation(self, include_warning: bool) -> DataPointT5Representation:

        if include_warning:
            inputs = (
                f"fix "
                f"{self.rule_report.rule_id} "
                f"{self.rule_report.message} {self.warning_line}:\n"
                f"{self.source_code} </s>"
            )
        else:
            inputs = f"fix {self.source_code} </s>"
        outputs = self.target_code + " </s>"
        return DataPointT5Representation(input=inputs, output=outputs)


def get_data_as_python(data_json_path: str) -> List[DataPoint]:
    # converts a data point in json format to a data point in python object
    def from_json_to_python(sample: Dict[str, Any]) -> DataPoint:
        rule_report = RuleReport(
            sample["rule_report"]["rule_id"],
            sample["rule_report"]["message"],
            sample["rule_report"]["col_begin"],
            sample["rule_report"]["col_end"],
            sample["rule_report"]["line_begin"],
            sample["rule_report"]["line_end"],
            sample["rule_report"]["severity"],
        )
        instructions = []
        for inst in sample["instructions"]:
            instruction = Instruction(
                inst["type"],
                inst["text"],
                inst["line_number"],
                inst["line_column"],
                inst["global_idx"],
                inst["description"],
            )
            instructions.append(instruction)

        data_point = DataPoint(
            sample["source_code"],
            sample["target_code"],
            sample["warning_line"],
            rule_report,
            instructions,
            sample["initial_source_patch"],
            sample["initial_target_patch"],
            sample["repo"],
            sample["source_filename"],
            sample["target_filename"],
            sample["source_changeid"],
            sample["target_changeid"],
        )
        return data_point

    with open(data_json_path, "r", errors="ignore") as f:
        data_json = json.load(f)
    data = [from_json_to_python(sample) for sample in data_json]
    return data


def get_data_from_paths(paths: List[str]) -> List[DataPoint]:
    data = []
    for path in paths:
        current_data = get_data_as_python(path)
        data += current_data
    return data
