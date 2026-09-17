"""Validation and freezing for model-generated feature schemas."""

import hashlib
import json
import re


SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SENSITIVE_TERMS = {
    "age", "gender", "sex", "race", "ethnicity", "identity", "name",
    "occupation", "job", "health", "disease", "disability", "emotion",
    "personality", "attractiveness", "income", "religion", "nationality",
    "年龄", "性别", "种族", "民族", "身份", "职业", "健康", "残疾", "情绪",
    "性格", "美貌", "收入", "宗教",
}


def _contains_sensitive(value):
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(term.lower() in text for term in SENSITIVE_TERMS)


def _rule_feature_ids(value, key=""):
    references = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name in {"feature_id", "if_feature", "then_feature", "source_feature", "target_feature", "depends_on"}:
                if isinstance(child, str):
                    references.append(child)
                elif isinstance(child, list):
                    references.extend(item for item in child if isinstance(item, str))
            references.extend(_rule_feature_ids(child, name))
    elif isinstance(value, list):
        for child in value:
            references.extend(_rule_feature_ids(child, key))
    return references


def _validate_feature(item, index, feature_ids, policy):
    if not isinstance(item, dict):
        raise ValueError(f"feature_{index}_not_object")
    required = {"feature_id", "group", "value_type", "possible_values", "allow_unknown", "evidence_required"}
    optional = {"exclusive_values"}
    if not required.issubset(item) or not set(item).issubset(required | optional):
        raise ValueError(f"feature_{index}_fields_mismatch")
    feature_id = item["feature_id"]
    if not isinstance(feature_id, str) or not SNAKE_CASE.fullmatch(feature_id):
        raise ValueError(f"feature_{index}_invalid_id")
    if feature_id in feature_ids:
        raise ValueError("duplicate_feature_id")
    feature_ids.add(feature_id)
    if _contains_sensitive(item):
        raise ValueError(f"sensitive_feature:{feature_id}")
    if not isinstance(item["group"], str) or not item["group"].strip():
        raise ValueError(f"feature_{index}_invalid_group")
    value_type = item["value_type"]
    allowed_types = set(policy.get("value_types", ["enum", "set", "number", "integer"]))
    if value_type not in allowed_types:
        raise ValueError(f"feature_{index}_invalid_value_type")
    values = item["possible_values"]
    if value_type in {"enum", "set"}:
        if not isinstance(values, list) or not values:
            raise ValueError(f"feature_{index}_invalid_possible_values")
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"feature_{index}_invalid_possible_values")
        if len(values) != len(set(values)):
            raise ValueError(f"feature_{index}_invalid_possible_values")
        if "unknown" not in values:
            raise ValueError(f"feature_{index}_missing_unknown")
        if value_type == "set" and item.get("exclusive_values"):
            if not isinstance(item["exclusive_values"], list) or not set(item["exclusive_values"]).issubset(values):
                raise ValueError(f"feature_{index}_invalid_exclusive_values")
    elif values != []:
        raise ValueError(f"feature_{index}_numeric_values_must_be_empty")
    if item["allow_unknown"] is not True or item["evidence_required"] is not True:
        raise ValueError(f"feature_{index}_strict_flags_required")
    return item


def validate_schema(schema, policy=None):
    policy = policy or {}
    if not isinstance(schema, dict):
        raise ValueError("schema_not_object")
    required_schema_fields = {"schema_version", "category", "features", "rules"}
    if not required_schema_fields.issubset(schema) or not set(schema).issubset(required_schema_fields | {"schema_fingerprint"}):
        raise ValueError("schema_fields_mismatch")
    if not isinstance(schema["schema_version"], str) or not schema["schema_version"].strip():
        raise ValueError("invalid_schema_version")
    if not isinstance(schema["category"], str) or not schema["category"].strip():
        raise ValueError("invalid_schema_category")
    features = schema["features"]
    if not isinstance(features, list):
        raise ValueError("features_not_array")
    minimum = int(policy.get("min_features", 8))
    maximum = int(policy.get("max_features", 30))
    if not minimum <= len(features) <= maximum:
        raise ValueError("feature_count_out_of_range")
    feature_ids = set()
    for index, item in enumerate(features):
        _validate_feature(item, index, feature_ids, policy)
    required_ids = set(policy.get("required_feature_ids", []))
    missing_required = required_ids - feature_ids if schema["category"] == "person" else set()
    if missing_required:
        raise ValueError("required_features_missing:" + ",".join(sorted(missing_required)))
    if not isinstance(schema["rules"], list) or any(not isinstance(rule, dict) for rule in schema["rules"]):
        raise ValueError("invalid_rules")
    if any(reference not in feature_ids for rule in schema["rules"] for reference in _rule_feature_ids(rule)):
        raise ValueError("rule_references_unknown_feature")
    if "schema_fingerprint" in schema and schema["schema_fingerprint"] != fingerprint_schema(schema):
        raise ValueError("schema_fingerprint_mismatch")
    if _contains_sensitive(schema):
        raise ValueError("sensitive_schema")
    return schema


def fingerprint_schema(schema):
    """Hash semantic schema content, excluding its generated version label."""
    semantic = {key: schema[key] for key in ("category", "features", "rules")}
    payload = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_schema(schema, policy=None):
    validate_schema(schema, policy)
    fingerprint = fingerprint_schema(schema)
    frozen = dict(schema)
    frozen["schema_version"] = f"auto_{schema['category']}_{fingerprint[:12]}"
    frozen["schema_fingerprint"] = fingerprint
    validate_schema({key: frozen[key] for key in ("schema_version", "category", "features", "rules")}, policy)
    return frozen


def feature_output_schema(schema):
    properties = {}
    required = []
    for item in schema["features"]:
        kind = item["value_type"]
        if kind == "enum":
            definition = {"type": "string", "enum": item["possible_values"]}
        elif kind == "set":
            definition = {"type": "array", "items": {"type": "string", "enum": item["possible_values"]}, "uniqueItems": True}
        elif kind == "integer":
            definition = {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["unknown"]}]}
        else:
            definition = {"anyOf": [{"type": "number"}, {"type": "string", "enum": ["unknown"]}]}
        properties[item["feature_id"]] = definition
        required.append(item["feature_id"])
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "image_id": {"type": "string"},
            "target_instance_id": {"type": "string"},
            "features": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            "evidence": {"type": "object", "properties": {key: {"type": "string"} for key in required}, "required": required, "additionalProperties": False},
        },
        "required": ["schema_version", "image_id", "target_instance_id", "features", "evidence"],
        "additionalProperties": False,
    }


def validate_feature_result(result, row, schema):
    expected = {item["feature_id"] for item in schema["features"]}
    if result.get("schema_version") != schema["schema_version"]:
        raise ValueError("feature_schema_version_mismatch")
    if result.get("image_id") != row["image_id"] or result.get("target_instance_id") != row["target_instance_id"]:
        raise ValueError("feature_identity_mismatch")
    features = result.get("features")
    if not isinstance(features, dict) or set(features) != expected:
        raise ValueError("feature_fields_mismatch")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise ValueError("evidence_fields_mismatch")
    definitions = {item["feature_id"]: item for item in schema["features"]}
    for key, value in features.items():
        item = definitions[key]
        if item["value_type"] == "enum" and value not in item["possible_values"]:
            raise ValueError(f"invalid_feature_value:{key}")
        if item["value_type"] == "set":
            if not isinstance(value, list) or not value or len(value) != len(set(value)) or any(v not in item["possible_values"] for v in value):
                raise ValueError(f"invalid_feature_value:{key}")
            if any(v in value for v in item.get("exclusive_values", [])) and len(value) != 1:
                raise ValueError(f"invalid_feature_value:{key}")
        if item["value_type"] == "integer" and value != "unknown" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"invalid_feature_value:{key}")
        if item["value_type"] == "number" and value != "unknown" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"invalid_feature_value:{key}")
        if not isinstance(evidence[key], str) or not evidence[key].strip():
            raise ValueError(f"invalid_evidence:{key}")
    return result
