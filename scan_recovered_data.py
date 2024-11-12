import uuid
import warnings
from datetime import datetime
from pathlib import PurePath, Path
from typing import Optional, List, Tuple, Union

import datasets
from dataclasses import dataclass, asdict


@dataclass
class SearchFile:
    file_name: str
    header: Optional[str] = None
    footer: Optional[str] = None
    dump_files: Union[List[str], Tuple[str]] = ()
    max_length: int = 1000
    footer_slack: int = 20

    def __post_init__(self):
        assert not (self.header is None and self.footer is None), "At least one of header or footer needs to be defined but both were None!"
        assert self.max_length > 0, "Each file version should have at least one line!"


def dump_file_version(lines: List[str], save_path: str, force_overwrite: bool = False):
    if isinstance(lines, str):
        warnings.warn("Expected lines to be a list of strings not a single string! Werapping it in a list and treat it as single line.")
    with open(save_path, 'a+' if not force_overwrite else 'w') as f:
        for line in lines:
            if not line.endswith('\n'):
                # insert missing newline at end of line
                line = line + '\n'
            f.write(line)


def scan_dump(example, save_directory: str, file_endings: str = '.py'):
    unique_worker_id = uuid.uuid4().hex
    file_dir = example['file_name']  # one directory per file name to collect all versions
    (Path(save_directory) / file_dir).mkdir(parents=True, exist_ok=False)
    # prepare header and footer line by line
    if example['header'] is not None:
        header_lines = example['header'].split('\n')
        header_lines = [line.strip() for line in header_lines]
    else:
        header_lines = []
    num_header_lines = len(header_lines)
    if example['footer'] is not None:
        footer_lines = example['footer'].split('\n')
        footer_lines = [line.strip() for line in footer_lines]
    else:
        footer_lines = []
    num_footer_lines = len(footer_lines)

    for file_dump in example['dump_files']:
        with open(file_dump, 'r', errors='ignore') as file:
            lines = file.readlines()
            print('num_lines', len(lines))
            # initialize all state bookkeeping variables
            header_match = 0
            header_start_line = None
            footer_match = 0
            footer_start_line = None
            line_number = 0
            file_lines = []
            second_pass_lines_to_extract = []
            fast_forward = None
            for line in lines:
                # ignore lines if they were marked for the second pass
                if fast_forward is not None and line_number <= fast_forward:
                    line_number += 1
                    continue
                elif fast_forward is not None and line_number > fast_forward:
                    fast_forward = None  # continue at the line after fast forward
                # check if header matches current line
                if header_match < num_header_lines:
                    if header_lines[header_match] in line.strip():
                        header_match += 1
                        if header_start_line is None:
                            header_start_line = line_number
                # check if footer matches current line
                if footer_match < num_footer_lines:
                    if footer_lines[footer_match] in line.strip():
                        footer_match += 1
                        if footer_start_line is None:
                            footer_start_line = line_number
                if header_match > 0:
                    file_lines.append(line)
                elif footer_match > 0:  # footer but no header -> save for second pass (no backtracking of lines)
                    file_end = footer_start_line + num_footer_lines + example['footer_slack']
                    file_start = max(0, file_end - example['max_length'] + 1)
                    second_pass_lines_to_extract.append((file_start, file_end))
                    fast_forward = file_end
                    # reset bookkeping state of file instance
                    header_match = 0
                    header_start_line = None
                    footer_match = 0
                    footer_start_line = None
                    file_lines = []
                    line_number += 1
                    continue
                if len(file_lines) > 0:
                    # if max length of a file version or the full footer + slack is reached dump the extracted lines
                    is_max_length = line_number - header_start_line + 1 >= example['max_length']
                    is_full_footer_match = footer_match == num_footer_lines
                    if_full_footer_slack = footer_start_line is not None and (footer_start_line + num_footer_lines + example['footer_slack']) == line_number
                    is_partial_match_at_file_end = len(file_lines) > 0 and line_number == (len(lines) - 1)
                    if is_max_length or (is_full_footer_match and if_full_footer_slack) or is_partial_match_at_file_end:
                        print('file_lines', len(file_lines), 'header_start_line', header_start_line, 'line_number', line_number, 'header_match', header_match, 'footer_match', footer_match)
                        file_name = f'{datetime.now().strftime("%Y%m%d_%H%M_%f")}_{header_start_line}_{header_start_line+len(file_lines)-1}_{unique_worker_id}{file_endings}'
                        save_path = str(PurePath(save_directory) / file_dir / file_name)
                        dump_file_version(file_lines, save_path)
                        # reset bookkeping state of file instance
                        header_match = 0
                        header_start_line = None
                        footer_match = 0
                        footer_start_line = None
                        file_lines = []
                line_number += 1
                # TODO think of end-of-file pattern (empty bytes -> weird characters) recognition and separate files based on that
            # TODO second pass
            if len(second_pass_lines_to_extract) > 0:
                line_number = 0
                second_pass_file = 0
                next_start_line, next_end_line = second_pass_lines_to_extract[second_pass_file]
                file_lines = []
                for line in lines:
                    if line_number < next_start_line:
                        line_number += 1
                        continue
                    elif line_number < next_end_line:
                        file_lines.append(line)
                    else:
                        file_lines.append(line)
                        file_name = f'{datetime.now().strftime("%Y%m%d_%H%M_%f")}_{next_start_line}_{next_end_line+len(file_lines)-1}_{unique_worker_id}{file_endings}'
                        save_path = str(PurePath(save_directory) / file_dir / file_name)
                        print('file_lines', len(file_lines), 'second_pass_file', second_pass_file, 'line_number', line_number)
                        dump_file_version(file_lines, save_path)
                        file_lines = []
                        second_pass_file += 1
                        if second_pass_file < len(second_pass_lines_to_extract):
                            next_start_line, next_end_line = second_pass_lines_to_extract[second_pass_file]
                        else:
                            break  # no more files to extract
                    line_number += 1


def main(save_directory='dump_scan'):
    search_files = [
        SearchFile(file_name='arguments', header=None, footer='/home/mamba/.cache/templates/standard_templates', dump_files=['dump_files/arguments_recovery.txt'], max_length=380, footer_slack=30),
        SearchFile(file_name='data_caching', header=None, footer='# object with to_huggingface() (only TableQuestionDataset?)', dump_files=['dump_files/data_caching_recovery.txt'], max_length=170, footer_slack=70),
        SearchFile(file_name='data_utils', header=None, footer='def plot_histogram(', dump_files=['dump_files/source_code_dump.txt'], max_length=400, footer_slack=80),
        SearchFile(file_name='tapas_model', header=None, footer='def tapas_generation(tokenizer, model_inputs: dict, model_outputs: dict, table: pd.DataFrame) -> List[str]:', dump_files=['dump_files/source_code_dump.txt'], max_length=350, footer_slack=200),
    ]

    search_file_dataset = datasets.Dataset.from_list([asdict(search_file) for search_file in search_files])

    # create save directory (error if already exists to not accidentaly overwrite stuff)
    Path(save_directory).mkdir(parents=True, exist_ok=False)

    search_file_dataset.map(
        scan_dump,
        fn_kwargs={
            'save_directory': save_directory,
        },
        num_proc=12,
    )


if __name__ == "__main__":
    main(save_directory='dump_scan_final_check_3')
