import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import json_lines
from dacite import from_dict
from ruamel.yaml import YAML
from tqdm import tqdm


def path(root, *path):
    return Path(root).joinpath(*path).expanduser().absolute()


@dataclass
class Load:
    dir: str
    jsonl: List[str]


@dataclass
class Dump:
    dir: str
    json: str
    split: str


@dataclass
class Range:
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass
class SeqLen:
    html: Optional[Range] = None
    cell: Optional[Range] = None


@dataclass
class TabNet:
    load: Load
    dump: Dump
    type: Literal["FinTabNet", "PubTabNet"]
    replace: Dict[Tuple, str]
    samples: Optional[int] = None
    seq_len: Optional[SeqLen] = None


class TabNetParser(object):
    def __init__(self, params: TabNet):
        self.params = params
        self.replace = set(self.params.replace.values())

    def splits(self):
        splits = defaultdict(list)
        for jsonl in self.params.load.jsonl:
            with json_lines.open(path(jsonl)) as reader:
                for item in tqdm(reader, desc=jsonl):
                    splits[item["split"]].append(item)
        return {k: v[: self.params.samples] for k, v in splits.items()}

    def merge(self, tokens):
        idx = 0
        merged = []
        while tokens[idx] not in ["</table>", "</tbody>"]:
            num = 2 if tokens[idx] == "<td>" else 1
            merged.append("".join(tokens[idx : idx + num]))
            idx += num
        merged.append(tokens[idx])
        return merged

    def insert_empty_cell(self, tokens, cells):
        cells = iter(cells)
        for token in tokens:
            if token == "<td></td>" or token == "<td":
                cell = next(cells)
                if "bbox" not in cell:
                    yield self.params.replace[tuple(cell["tokens"])]
                else:
                    yield token
            else:
                yield token

    def count_merged_tokens(self, tokens):
        targets = ["<td", "<td></td>", *self.replace]
        return sum([int(token in targets) for token in tokens])

    def get_thead_item_idx(self, tokens):
        if "<thead>" not in tokens:
            return range(0)
        return list(range(tokens[: tokens.index("</thead>")].count("</td>")))

    def remove_Bb(self, content):
        return list(tag for tag in content if tag != "<b>" and tag != "</b>")

    def is_long_sample(self, html, cell):
        if self.params.seq_len is None:
            return True

        if self.params.seq_len.html is not None:
            html_min = self.params.seq_len.html.min or 0
            html_max = self.params.seq_len.html.max or sys.maxsize
            if not (html_min <= len(html) <= html_max):
                return False

        if self.params.seq_len.cell is not None:
            cell_min = self.params.seq_len.cell.min or 0
            cell_max = self.params.seq_len.cell.max or sys.maxsize

            if not (cell_min <= len(cell) <= cell_max):
                return False

        return True

    def open_structure_file(self, split, item):
        name = os.path.splitext(item["filename"])[0]
        return open(path(self.params.dump.dir, split, f"{name}.txt"), "w")

    def format_cell(self, cell, is_head):
        if "bbox" not in cell:
            return "0,0,0,0<;><UKN>"
        else:
            bbox = ",".join([str(int(b)) for b in cell["bbox"]])
            text = cell["tokens"]
            if is_head:
                text = self.remove_Bb(text)
            text = "\t".join(text)
            return f"{bbox}<;>{text}" # ToDo: HTMLタグを除去する & ""を除去する

    def format_item(self, split, item, htmls, cells, f):
        print(path(self.params.load.dir, split, item["filename"]), file=f)
        marks = list(self.insert_empty_cell(self.merge(htmls), cells))
        heads = self.get_thead_item_idx(htmls)
        assert len(cells) == self.count_merged_tokens(marks)
        print(",".join(marks), file=f)
        for idx, cell in enumerate(cells):
            print(self.format_cell(cell, idx in heads), file=f)

    def parse_single(self, split, item):
        htmls = item["html"]["structure"]["tokens"]
        cells = item["html"]["cells"]
        if self.is_long_sample(htmls, cells):
            with self.open_structure_file(split, item) as f:
                self.format_item(split, item, htmls, cells, f)

            if split == params.dump.split:
                type = "complex" if ">" in htmls else "simple"
                full_text = self.insert_text_to_token(htmls, cells)
                return {item["filename"]: dict(html=full_text, type=type)}

        return {}

    def parse_all(self):
        test_data = {}
        for split, items in self.splits().items():
            os.makedirs(path(self.params.dump.dir, split), exist_ok=True)
            for item in tqdm(items, desc=split):
                test_data.update(self.parse_single(split, item))

        with open(path(self.params.dump.json), "w") as f:
            json.dump(test_data, f)

    def insert_text_to_token(self, html, cell):
        contents = iter(cell)
        restored = []
        for idx, el in enumerate(html):
            if el in ["<td></td>", "</td>"]:
                content = next(contents, None)
                if content is not None:
                    if params.type == "PubTabNet" or "bbox" in content:
                        content = "".join(content["tokens"])
                        el = el.replace("</", f"{content}</")
            restored.append(el)
        return "".join(restored)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args = args.parse_args()
    with open(path(args.config)) as f:
        params = from_dict(TabNet, YAML().load(f))
    TabNetParser(params).parse_all()
