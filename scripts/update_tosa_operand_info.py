import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

TOSA_SPEC_XML_SHA = "99c932000e54f0eb68e03129752a11e300e07695"  # v1.0
TOSA_SPEC_XML_URL = f"https://raw.githubusercontent.com/arm/tosa-specification/{TOSA_SPEC_XML_SHA}/tosa.xml"

SPIRV_TOSA_GRAMMAR_JSON_SHA = "7919b00b5f71bb3e6245c38c926501c009060602"
SPIRV_TOSA_GRAMMAR_JSON_URL = f"https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/{SPIRV_TOSA_GRAMMAR_JSON_SHA}/include/spirv/unified1/extinst.tosa.001000.1.grammar.json"
SUPPORTED_CATEGORIES = {
    "input",
    "output",
    "attribute",
    "attribute(pro-int,pro-fp)",
}

TOSA_OPERAND_INFO_PATH = Path(
    "src/vgf_adapter_model_explorer/resources/tosa_1_0_operand_info.json"
)


def fetch_url(url: str) -> bytes:
    """Fetch the contents of `url` and return raw bytes."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "model-explorer-vgf-info-gen/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def safe_fetch(url: str) -> bytes | None:
    try:
        return fetch_url(url)
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None


def must_get(mapping, key, what) -> Any:
    """Return `mapping[key]` if present and not None; otherwise raise KeyError/ValueError naming the missing `what`."""
    if key not in mapping:
        raise KeyError(f"Missing {what!r}: expected key '{key}' in {mapping}")

    val = mapping.get(key)
    if val is None:
        raise ValueError(
            f"Missing {what!r}: key '{key}' has None value in {mapping}"
        )
    return val


def parse_tosa_xml_enums(tosa_xml_bytes: bytes) -> Dict[str, List[str]]:
    """Parses the TOSA XML specification and return `{enum_name: [value_name, ...]}` for all `<enum>` blocks."""
    tosa_xml_root = ET.fromstring(tosa_xml_bytes)

    return {
        enum_name: [
            enum_value
            for ev in enum_element.findall("enumval")
            if (enum_value := must_get(ev.attrib, "name", "enum value"))
        ]
        for enum_element in tosa_xml_root.iterfind(".//enum")
        if (enum_name := must_get(enum_element.attrib, "name", "enum name"))
    }


def parse_tosa_xml_categories(
    tosa_xml_bytes: bytes,
) -> Dict[str, Dict[str, str]]:
    """Parses the TOSA XML specification to a category dictionary of `{OP_UPPER: {OPERAND_UPPER: category_lower}}`
    from `<operator>/<arguments>/<argument>`.

    In the TOSA specification (v1.0) these are called *arguments*. In the SPIR-V grammar
    and elsewhere in this code they are referred to as *operands*.
    """

    root: ET.Element = ET.fromstring(tosa_xml_bytes)
    category_map: Dict[str, Dict[str, str]] = {}
    for operator in root.findall(".//operator"):
        opname = (operator.findtext("name") or "").strip()
        if not opname:
            continue
        argument_categories: Dict[str, str] = {}
        for argument in operator.findall("arguments/argument"):
            argument_name = (argument.get("name") or "").strip()
            category = (argument.get("category") or "").strip().lower()
            if category not in SUPPORTED_CATEGORIES:
                raise ValueError(
                    f"Invalid category {category!r} on {opname!r} argument {argument_name}; "
                    f"expected one of {SUPPORTED_CATEGORIES}"
                )
            if argument_name and category:
                argument_categories[argument_name.upper()] = category
        category_map[opname.upper()] = argument_categories
    return category_map


def construct_tosa_operand_info(
    spirv_tosa_grammar_bytes: bytes, tosa_xml_bytes: bytes
) -> Dict[str, List[Dict[str, str]]]:
    """
    Join SPIR-V TOSA grammar with TOSA XML categories to produce
    `{normalized_opname: [{'name': operand_name, 'category': category}, ...]}`.
    Fails if any mapping is missing as that would
    imply a mismatch between the grammar and specification.
    Operator names are normalized due to MLIR having different casing.

    In the TOSA specification (v1.0) these are called *arguments*. In the SPIR-V grammar
    and elsewhere in this code they are referred to as *operands*.
    """

    category_map = parse_tosa_xml_categories(tosa_xml_bytes)
    grammar = json.loads(spirv_tosa_grammar_bytes.decode("utf-8"))
    instructions = grammar["instructions"]

    tosa_operand_info: Dict[str, List[Dict[str, str]]] = {}
    for ins in instructions:
        instruction_name = ins["opname"]
        normalised_ins_name = instruction_name.replace("_", "").lower()
        operand_categories = must_get(
            category_map, instruction_name.upper(), "operator"
        )
        rows: List[Dict[str, str]] = []
        for operand in ins["operands"]:
            operand_name = operand["name"]
            category = must_get(
                operand_categories,
                operand_name.upper(),
                f"operand {operand_name!r} of {instruction_name!r}",
            )

            rows.append({"name": operand_name, "category": category})
        tosa_operand_info[normalised_ins_name] = rows
    return tosa_operand_info


def main() -> int:
    """CLI entry: fetch pinned TOSA XML and SPIR-V TOSA grammar, build operand+enum info,
    and write JSON to the adapter repo's resources path. Returns process exit code."""

    tosa_xml_bytes, spirv_tosa_grammar_bytes = (
        safe_fetch(url)
        for url in [TOSA_SPEC_XML_URL, SPIRV_TOSA_GRAMMAR_JSON_URL]
    )

    if not (tosa_xml_bytes and spirv_tosa_grammar_bytes):
        return 1

    operand_info = construct_tosa_operand_info(
        spirv_tosa_grammar_bytes, tosa_xml_bytes
    )
    enum_map = parse_tosa_xml_enums(tosa_xml_bytes)

    TOSA_OPERAND_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOSA_OPERAND_INFO_PATH.open("w", encoding="utf-8") as info_file:
        json.dump(
            {"enums": enum_map, "operations": operand_info},
            info_file,
            sort_keys=True,
            indent=2,
        )
        info_file.write("\n")
    print(
        f"Wrote {TOSA_OPERAND_INFO_PATH} (ops={len(operand_info)}, enums={len(enum_map)})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
