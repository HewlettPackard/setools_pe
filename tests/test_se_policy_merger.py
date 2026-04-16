#!/usr/bin/env python3
"""
Comprehensive test suite for se_policy_merger.

All tests are self-contained — no SELinux tooling or live system required.

Run with markers to filter::

    pytest -m unit           # fast unit tests only
    pytest -m integration    # merger() and multi-file tests
    pytest -m cli            # subprocess CLI tests

Test categories
───────────────
  Unit           - individual functions: parsing, extraction, generation
  Regex          - compiled regex patterns against known inputs
  Merge          - integration tests through merger()
  Idempotency    - merger(output) == output
  Multi-file     - merging overlapping policies (deduplication)
  Counters       - verify summary counters
  discard_require - require-block stripping edge cases
  Real policies  - run against MYAPP_ALLOW_POLICY / MYAPP_BOOTSTRAP_POLICY
  CLI            - argparse and file I/O
  Edge / Error   - empty input, comments only, malformed lines
"""

import os
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest  # pyright: ignore[reportMissingImports]

from conftest import se_policy_merger as spm, _PROJECT_ROOT

_SCRIPT_PATH = os.path.join(_PROJECT_ROOT, "se_policy_merger")


# ── helper ──────────────────────────────────────────────────────────────────
def _lines(text):
    """Split a string into a list of lines (each ending with \\n)."""
    return [l + "\n" for l in text.splitlines()]


# ── sample policies ────────────────────────────────────────────────────────
MINIMAL_ALLOW = textwrap.dedent("""\
    module myapp 1.0;

    allow myapp_t tmp_t:file read;
""")

MULTI_PERM_ALLOW = textwrap.dedent("""\
    module myapp 1.0;

    allow myapp_t tmp_t:file { read write getattr };
""")

SINGLE_PERM_ALLOW = textwrap.dedent("""\
    module myapp 1.0;

    allow init_t myapp_t:process transition;
""")

ROLE_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    role system_r types { myapp_t helper_t };
    role unconfined_r types myapp_t;
""")

TYPE_TRANSITION_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type_transition init_t myapp_exec_t:process myapp_t;
    type_transition unconfined_t myapp_exec_t:process myapp_t;
""")

TYPE_DEFINITION_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;
    type myapp_exec_t;
    type myapp_file_t;
""")

TYPEATTRIBUTE_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    typeattribute myapp_exec_t file_type;
    typeattribute myapp_file_t file_type, domain;
""")

ROLE_TRANSITION_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    role_transition unconfined_r myapp_exec_t system_r;
""")

REQUIRE_BLOCK_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type init_t, tmp_t;
        role system_r;
        attribute file_type;
        class file { read write };
    }

    allow myapp_t init_t:file read;
    allow myapp_t tmp_t:file write;
""")

FULL_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;
    type myapp_exec_t;
    type myapp_file_t;

    require {
        type init_t, unconfined_t, tmp_t;
        role system_r, unconfined_r;
        attribute file_type;
        class file { read write entrypoint execute };
        class process { transition };
        class dir { search read };
    }

    role system_r types { myapp_t };
    role_transition unconfined_r myapp_exec_t system_r;
    type_transition init_t myapp_exec_t:process myapp_t;
    typeattribute myapp_exec_t file_type;
    allow myapp_t init_t:file { read write };
    allow myapp_t tmp_t:dir { search read };
    allow init_t myapp_exec_t:file { entrypoint execute };
    allow init_t myapp_t:process transition;
""")

# Two overlapping policies for merge testing
POLICY_A = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type init_t, tmp_t;
        class file { read write };
    }

    allow myapp_t init_t:file read;
    allow myapp_t tmp_t:file write;
""")

POLICY_B = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;
    type myapp_exec_t;

    require {
        type init_t, bin_t;
        class file { read execute };
    }

    allow myapp_t init_t:file { read execute };
    allow myapp_t bin_t:file read;
""")

# ── synthetic "realistic" policies (replace myapp.te / myapp_bootstrap.te) ─────
# MYAPP_BOOTSTRAP_POLICY: bootstrap declarations — types, roles, transitions,
# typeattributes, but no allow rules (mirrors the real myapp_bootstrap.te).
MYAPP_BOOTSTRAP_POLICY = textwrap.dedent("""\
    module myapp 0.1;

    type myapp_exec_t;
    type myapp_t;
    type myapp_file_t;
    type myapp_port_t;

    require {
        type myapp_t, myapp_exec_t, myapp_file_t, myapp_port_t;
        type init_t, unconfined_t;
        role system_r, unconfined_r;
        attribute file_type, port_type;
        class process { transition };
    }

    role system_r types { myapp_t };
    type_transition init_t myapp_exec_t:process myapp_t;
    type_transition unconfined_t myapp_exec_t:process myapp_t;
    role_transition unconfined_r myapp_exec_t system_r;
    typeattribute myapp_exec_t file_type;
    typeattribute myapp_file_t file_type;
    typeattribute myapp_port_t port_type;
""")

# MYAPP_ALLOW_POLICY: allow rules that complement the bootstrap (mirrors myapp.te).
MYAPP_ALLOW_POLICY = textwrap.dedent("""\
    module myapp 0.1;

    type myapp_t;
    type myapp_exec_t;
    type myapp_file_t;

    require {
        type myapp_t, myapp_exec_t, myapp_file_t;
        type init_t, tmp_t, bin_t;
        role system_r;
        class file { read write getattr execute entrypoint };
        class dir { search read open };
        class process { transition };
    }

    allow myapp_t tmp_t:file { read write };
    allow myapp_t myapp_file_t:file { read write getattr };
    allow myapp_t bin_t:file { read execute };
    allow init_t myapp_exec_t:file { entrypoint execute };
    allow init_t myapp_t:process transition;
    allow myapp_t tmp_t:dir { search read open };
""")


# ── autouse fixture: reset global state before each test ────────────────────
@pytest.fixture(autouse=True)
def _clean_state():
    """Ensure every test starts from a clean global state."""
    spm.reset_state()
    yield
    spm.reset_state()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — read_file
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestReadFile:
    def test_reads_file_lines(self, tmp_path):
        """read_file returns a list of lines with newlines preserved."""
        p = tmp_path / "policy.te"
        p.write_text("module myapp 1.0;\nallow a_t b_t:file read;\n")
        result = spm.read_file(str(p))
        assert result == ["module myapp 1.0;\n", "allow a_t b_t:file read;\n"]

    def test_nonexistent_file_raises(self):
        """read_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            spm.read_file("/nonexistent/path/policy.te")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — reset_state
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestResetState:
    def test_clears_rules(self):
        spm.rules["allow"].append("allow a b:c d;")
        spm.reset_state()
        assert all(len(v) == 0 for v in spm.rules.values())

    def test_clears_dicts_and_sets(self):
        spm.allow_rules["x"] = {}
        spm.required_types.add("x_t")
        spm.type_definition.add("y_t")
        spm.required_classes["file"] = {"read"}
        spm.reset_state()
        assert len(spm.allow_rules) == 0
        assert len(spm.required_types) == 0
        assert len(spm.type_definition) == 0
        assert len(spm.required_classes) == 0

    def test_resets_counters(self):
        spm.allow_counter = 42
        spm.role_counter = 7
        spm.reset_state()
        assert spm.allow_counter == 0
        assert spm.role_counter == 0

    def test_resets_module_title(self):
        spm.MODULE_TITLE = "custom 2.0"
        spm.reset_state()
        assert spm.MODULE_TITLE == "myapp 1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — discard_require
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDiscardRequire:
    def test_removes_simple_require(self):
        lines = _lines(REQUIRE_BLOCK_POLICY)
        result = spm.discard_require(lines)
        joined = "".join(result)
        assert "require {" not in joined
        assert "type init_t" not in joined
        # rules should survive
        assert "allow myapp_t init_t:file read;" in joined

    def test_keeps_non_require_content(self):
        lines = _lines("module myapp 1.0;\ntype foo_t;\nallow foo_t bar_t:file read;\n")
        result = spm.discard_require(lines)
        joined = "".join(result)
        assert "module myapp 1.0;" in joined
        assert "type foo_t;" in joined
        assert "allow foo_t bar_t:file read;" in joined

    def test_no_require_returns_unchanged(self):
        lines = _lines(MINIMAL_ALLOW)
        result = spm.discard_require(lines)
        assert result == lines

    def test_multiple_require_blocks(self):
        doc = textwrap.dedent("""\
            require {
                type a_t;
            }
            allow a_t b_t:file read;
            require {
                type c_t;
            }
            allow c_t d_t:file write;
        """)
        result = spm.discard_require(_lines(doc))
        joined = "".join(result)
        assert "require" not in joined
        assert "allow a_t b_t:file read;" in joined
        assert "allow c_t d_t:file write;" in joined

    def test_preserves_content_after_closing_brace(self):
        lines = ["require { type a_t; }allow x_t y_t:file read;\n"]
        result = spm.discard_require(lines)
        joined = "".join(result)
        assert "allow x_t y_t:file read;" in joined

    def test_empty_require_block(self):
        result = spm.discard_require(_lines("require {}\n"))
        # Block removed, nothing left but possibly empty lines
        for line in result:
            assert "require" not in line

    def test_nested_braces_in_require(self):
        """Require block with nested braces (e.g. class perm sets) is fully removed."""
        doc = textwrap.dedent("""\
            require {
                type a_t;
                class file { read write };
            }
            allow a_t b_t:file read;
        """)
        result = spm.discard_require(_lines(doc))
        joined = "".join(result)
        assert "require" not in joined
        assert "class file" not in joined
        assert "allow a_t b_t:file read;" in joined

    def test_negative_brace_compensation(self):
        """A stray '}' after require close is re-inserted to avoid losing outer braces."""
        # Simulate: require block closes, then an extra '}' appears on the same line
        # "require { type a_t; }}rest\n" -> brace goes to -1, should re-insert one '}'
        lines = ["require { type a_t; }}rest\n"]
        result = spm.discard_require(lines)
        joined = "".join(result)
        assert "}" in joined
        assert "rest" in joined


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — sort_rules_by_type
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSortRulesByType:
    def test_categorizes_allow(self):
        spm.sort_rules_by_type(_lines("allow myapp_t tmp_t:file read;\n"))
        assert len(spm.rules["allow"]) == 1

    def test_categorizes_role(self):
        spm.sort_rules_by_type(_lines("role system_r types { myapp_t };\n"))
        assert len(spm.rules["role"]) == 1

    def test_categorizes_type_transition(self):
        spm.sort_rules_by_type(_lines("type_transition init_t myapp_exec_t:process myapp_t;\n"))
        assert len(spm.rules["type_transition"]) == 1

    def test_categorizes_type_definition(self):
        spm.sort_rules_by_type(_lines("type myapp_t;\n"))
        assert len(spm.rules["type"]) == 1

    def test_categorizes_typeattribute(self):
        spm.sort_rules_by_type(_lines("typeattribute myapp_t file_type;\n"))
        assert len(spm.rules["typeattribute"]) == 1

    def test_categorizes_role_transition(self):
        spm.sort_rules_by_type(_lines("role_transition unconfined_r myapp_exec_t system_r;\n"))
        assert len(spm.rules["role_transition"]) == 1

    def test_module_line_sets_title(self):
        spm.sort_rules_by_type(_lines("module myapp 1.1;\n"))
        assert spm.MODULE_TITLE == "myapp 1.1"

    def test_decorative_comments_skipped(self):
        spm.sort_rules_by_type(_lines("######## ALLOW RULES ########\n"))
        assert all(len(v) == 0 for v in spm.rules.values())

    def test_unknown_line_warns(self, capsys):
        spm.sort_rules_by_type(_lines("unknown_keyword foo bar;\n"))
        err = capsys.readouterr().err
        assert "unable to categorize" in err

    def test_empty_lines_ignored(self, capsys):
        spm.sort_rules_by_type(["\n", "  \n", "\n"])
        err = capsys.readouterr().err
        assert "error" not in err.lower()

    def test_non_decorative_comment_logged_with_show_info(self, capsys):
        """Non-decorative comments are reported on stderr when show_info is True."""
        spm.show_info = True
        try:
            spm.sort_rules_by_type(_lines("# a simple policy comment\n"))
            err = capsys.readouterr().err
            assert "commented line ignored" in err
        finally:
            spm.show_info = False

    def test_leading_whitespace_categorized(self):
        """Lines with leading whitespace are stripped and categorized by their first word."""
        spm.sort_rules_by_type(_lines("  allow myapp_t tmp_t:file read;\n"))
        assert len(spm.rules["allow"]) == 1
        # The stored line should be stripped so downstream regex parsing works
        assert spm.rules["allow"][0] == "allow myapp_t tmp_t:file read;"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — extract_from_* (parsing)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExtractFromAllowRule:
    def test_multi_perm(self):
        spm.extract_from_allow_rule("allow myapp_t tmp_t:file { read write getattr };")
        assert "myapp_t" in spm.allow_rules
        assert "tmp_t" in spm.allow_rules["myapp_t"]
        perms = spm.allow_rules["myapp_t"]["tmp_t"]["file"]
        assert perms == {"read", "write", "getattr"}

    def test_single_perm(self):
        spm.extract_from_allow_rule("allow init_t myapp_t:process transition;")
        assert "init_t" in spm.allow_rules
        perms = spm.allow_rules["init_t"]["myapp_t"]["process"]
        assert perms == {"transition"}

    def test_populates_required_types(self):
        spm.extract_from_allow_rule("allow a_t b_t:file read;")
        assert "a_t" in spm.required_types
        assert "b_t" in spm.required_types

    def test_populates_required_classes(self):
        spm.extract_from_allow_rule("allow a_t b_t:dir { search read };")
        assert "dir" in spm.required_classes
        assert "search" in spm.required_classes["dir"]
        assert "read" in spm.required_classes["dir"]

    def test_merges_perms_same_rule(self):
        spm.extract_from_allow_rule("allow a_t b_t:file { read };")
        spm.extract_from_allow_rule("allow a_t b_t:file { write };")
        perms = spm.allow_rules["a_t"]["b_t"]["file"]
        assert perms == {"read", "write"}

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_allow_rule("allow totally broken")
        err = capsys.readouterr().err
        assert "error parsing" in err


@pytest.mark.unit
class TestExtractFromRoleRule:
    def test_multi_type(self):
        spm.extract_from_role_rule("role system_r types { myapp_t helper_t };")
        assert "system_r" in spm.role_rules
        assert spm.role_rules["system_r"] == {"myapp_t", "helper_t"}
        assert "system_r" in spm.required_roles

    def test_single_type(self):
        spm.extract_from_role_rule("role unconfined_r types myapp_t;")
        assert spm.role_rules["unconfined_r"] == {"myapp_t"}

    def test_merges_types(self):
        spm.extract_from_role_rule("role system_r types { myapp_t };")
        spm.extract_from_role_rule("role system_r types { helper_t };")
        assert spm.role_rules["system_r"] == {"myapp_t", "helper_t"}

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_role_rule("role broken")
        err = capsys.readouterr().err
        assert "error parsing" in err


@pytest.mark.unit
class TestExtractFromTypeTransition:
    def test_basic(self):
        spm.extract_from_type_transition(
            "type_transition init_t myapp_exec_t:process myapp_t;"
        )
        assert spm.type_transition_rules["init_t"]["myapp_exec_t"]["process"] == "myapp_t"
        assert "init_t" in spm.required_types
        assert "myapp_exec_t" in spm.required_types
        assert "myapp_t" in spm.required_types

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_type_transition("type_transition broken;")
        err = capsys.readouterr().err
        assert "error parsing" in err

    def test_populates_required_classes_with_transition(self):
        """When the class is not yet in required_classes, it should be seeded with {'transition'}."""
        spm.extract_from_type_transition(
            "type_transition init_t myapp_exec_t:process myapp_t;"
        )
        assert "process" in spm.required_classes
        assert "transition" in spm.required_classes["process"]

    def test_conflicting_final_type_last_wins(self):
        """When two type_transition rules share source+target+class, the last one wins."""
        spm.extract_from_type_transition(
            "type_transition init_t exec_t:process first_t;"
        )
        spm.extract_from_type_transition(
            "type_transition init_t exec_t:process second_t;"
        )
        assert spm.type_transition_rules["init_t"]["exec_t"]["process"] == "second_t"
        # Both final types should still be tracked for the require block
        assert "first_t" in spm.required_types
        assert "second_t" in spm.required_types


@pytest.mark.unit
class TestExtractFromTypeDefinition:
    def test_single(self):
        spm.extract_from_type_definition("type myapp_t;")
        assert "myapp_t" in spm.type_definition

    def test_multi_comma(self):
        spm.extract_from_type_definition("type myapp_t, myapp_exec_t;")
        assert "myapp_t" in spm.type_definition
        assert "myapp_exec_t" in spm.type_definition

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_type_definition("type ;")
        # The regex requires at least one \w/,/space char after 'type '; ';' alone fails
        err = capsys.readouterr().err
        assert "error parsing" in err


@pytest.mark.unit
class TestExtractFromTypeattribute:
    def test_single_attr(self):
        spm.extract_from_typeattribute_statement("typeattribute myapp_exec_t file_type;")
        assert "myapp_exec_t" in spm.typeattribute_rules
        assert "file_type" in spm.typeattribute_rules["myapp_exec_t"]
        assert "file_type" in spm.required_attributes

    def test_multi_attr(self):
        spm.extract_from_typeattribute_statement(
            "typeattribute myapp_t file_type, domain;"
        )
        assert spm.typeattribute_rules["myapp_t"] == {"file_type", "domain"}
        assert "file_type" in spm.required_attributes
        assert "domain" in spm.required_attributes

    def test_merges_attrs(self):
        spm.extract_from_typeattribute_statement("typeattribute x_t file_type;")
        spm.extract_from_typeattribute_statement("typeattribute x_t domain;")
        assert spm.typeattribute_rules["x_t"] == {"file_type", "domain"}

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_typeattribute_statement("typeattribute broken")
        err = capsys.readouterr().err
        assert "error parsing" in err

    def test_populates_required_types(self):
        """The type itself should be added to required_types."""
        spm.extract_from_typeattribute_statement("typeattribute myapp_t file_type;")
        assert "myapp_t" in spm.required_types


@pytest.mark.unit
class TestExtractFromRoleTransition:
    def test_basic(self):
        spm.extract_from_role_transition(
            "role_transition unconfined_r myapp_exec_t system_r;"
        )
        assert "unconfined_r" in spm.role_transition_rules
        assert "system_r" in spm.role_transition_rules["unconfined_r"]
        assert "myapp_exec_t" in spm.role_transition_rules["unconfined_r"]["system_r"]
        assert "unconfined_r" in spm.required_roles
        assert "system_r" in spm.required_roles
        assert "myapp_exec_t" in spm.required_types

    def test_malformed_prints_error(self, capsys):
        spm.extract_from_role_transition("role_transition broken;")
        err = capsys.readouterr().err
        assert "error parsing" in err


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — create_* (output generation)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCreateRequiredTypes:
    def test_sorted_output(self):
        spm.required_types.update(["b_t", "a_t", "c_t"])
        result = spm.create_required_types()
        assert result == "type a_t, b_t, c_t;\n"

    def test_self_removed(self):
        spm.required_types.update(["a_t", "self"])
        result = spm.create_required_types()
        assert "self" not in result
        assert "a_t" in result

    def test_with_indent(self):
        spm.required_types.add("a_t")
        result = spm.create_required_types("    ")
        assert result.startswith("    type")

    def test_empty_set_produces_bare_type_line(self):
        """With no required types, create_required_types produces 'type ;\\n' (invalid but guarded by merger)."""
        result = spm.create_required_types()
        assert result == "type ;\n"


@pytest.mark.unit
class TestCreateRequiredRoles:
    def test_sorted(self):
        spm.required_roles.update(["unconfined_r", "system_r"])
        result = spm.create_required_roles()
        assert result == "role system_r, unconfined_r;\n"

    def test_empty_set(self):
        """With no required roles, documents the bare output."""
        result = spm.create_required_roles()
        assert result == "role ;\n"


@pytest.mark.unit
class TestCreateRequiredAttributes:
    def test_sorted(self):
        spm.required_attributes.update(["port_type", "file_type"])
        result = spm.create_required_attributes()
        assert result == "attribute file_type, port_type;\n"

    def test_empty_set(self):
        """With no required attributes, documents the bare output."""
        result = spm.create_required_attributes()
        assert result == "attribute ;\n"


@pytest.mark.unit
class TestCreateRequiredClasses:
    def test_sorted_perms(self):
        spm.required_classes["file"] = {"write", "read", "append"}
        result = spm.create_required_classes()
        assert "class file { append read write };" in result

    def test_sorted_classes(self):
        spm.required_classes["dir"] = {"search"}
        spm.required_classes["file"] = {"read"}
        result = spm.create_required_classes()
        lines = result.strip().split("\n")
        # dir should come before file alphabetically
        assert "class dir" in lines[0]
        assert "class file" in lines[1]


@pytest.mark.unit
class TestCreateTypeDefinition:
    def test_sorted(self):
        spm.type_definition.update(["z_t", "a_t", "m_t"])
        result = spm.create_type_definition()
        lines = result.strip().split("\n")
        assert lines[0] == "type a_t;"
        assert lines[1] == "type m_t;"
        assert lines[2] == "type z_t;"
        assert spm.type_definition_counter == 3

    def test_with_indent(self):
        spm.type_definition.add("x_t")
        result = spm.create_type_definition("  ")
        assert result.startswith("  type x_t;")


@pytest.mark.unit
class TestCreateAllowRules:
    def test_sorted_perms(self):
        spm.allow_rules["myapp_t"] = {"tmp_t": {"file": {"write", "read", "append"}}}
        result = spm.create_allow_rules()
        assert "allow myapp_t tmp_t:file { append read write };" in result
        assert spm.allow_counter == 3

    def test_contains_header(self):
        spm.allow_rules["a_t"] = {"b_t": {"file": {"read"}}}
        result = spm.create_allow_rules()
        assert "ALLOW RULES" in result

    def test_subject_separators(self):
        spm.allow_rules["a_t"] = {"x_t": {"file": {"read"}}}
        spm.allow_rules["b_t"] = {"y_t": {"file": {"write"}}}
        result = spm.create_allow_rules()
        # create_allow_rules always groups by subject via make_it_shine
        assert "######## a_t ########" in result
        assert "######## b_t ########" in result


@pytest.mark.unit
class TestCreateRoleRules:
    def test_sorted_types(self):
        spm.role_rules["system_r"] = {"helper_t", "myapp_t"}
        result = spm.create_role_rules()
        assert "role system_r types { helper_t myapp_t };" in result
        assert spm.role_counter == 2


@pytest.mark.unit
class TestCreateTypeTransitionRules:
    def test_basic(self):
        spm.type_transition_rules["init_t"] = {"myapp_exec_t": {"process": "myapp_t"}}
        result = spm.create_type_transition_rules()
        assert "type_transition init_t myapp_exec_t:process myapp_t;" in result
        assert spm.type_transition_counter == 1
        assert "TYPE TRANSITION RULES" in result

    def test_pattern_grouping_for_large_sets(self):
        """When > 10 rules exist the function switches to per-subject separators."""
        for i in range(11):
            spm.type_transition_rules[f"src_{i}_t"] = {
                f"exec_{i}_t": {"process": f"app_{i}_t"}
            }
        result = spm.create_type_transition_rules()
        assert spm.type_transition_counter == 11
        # Subject separator should appear for at least the first source type
        assert "######## src_0_t ########" in result


@pytest.mark.unit
class TestCreateRoleTransitionRules:
    def test_basic(self):
        spm.role_transition_rules["unconfined_r"] = {"system_r": {"myapp_exec_t"}}
        result = spm.create_role_transition_rules()
        assert "role_transition unconfined_r myapp_exec_t system_r;" in result
        assert spm.role_transition_counter == 1

    def test_pattern_grouping_for_large_sets(self):
        """When > 10 rules exist the function switches to per-subject separators."""
        for i in range(11):
            spm.role_transition_rules[f"role_{i}_r"] = {"system_r": {f"exec_{i}_t"}}
        result = spm.create_role_transition_rules()
        assert spm.role_transition_counter == 11
        assert "######## role_0_r ########" in result


@pytest.mark.unit
class TestCreateTypeattributeStatement:
    def test_sorted_attrs(self):
        spm.typeattribute_rules["myapp_t"] = {"file_type", "domain"}
        result = spm.create_typeattribute_statement()
        assert "typeattribute myapp_t domain, file_type;" in result
        assert spm.typeattribute_counter == 2

    def test_pattern_grouping_for_large_sets(self):
        """When > 10 rules exist the function switches to per-subject separators."""
        for i in range(11):
            spm.typeattribute_rules[f"type_{i}_t"] = {"file_type"}
        result = spm.create_typeattribute_statement()
        assert spm.typeattribute_counter == 11
        assert "######## type_0_t ########" in result


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — make_it_shine
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMakeItShine:
    def test_no_pattern(self):
        lines = ["allow a_t b_t:file read;", "allow c_t d_t:file write;"]
        result = spm.make_it_shine(lines, None, "TEST SECTION")
        assert "TEST SECTION" in result
        assert "allow a_t" in result
        assert "allow c_t" in result

    def test_with_pattern_groups_by_subject(self):
        lines = [
            "allow a_t x_t:file read;",
            "allow a_t y_t:file write;",
            "allow b_t z_t:file read;",
        ]
        result = spm.make_it_shine(lines, spm.REGEX_ALLOW_TEMPLATE, "ALLOW RULES")
        # Each unique subject gets its own separator line: '######## <subject> ########'
        assert "######## a_t ########" in result
        assert "######## b_t ########" in result
        # a_t separator precedes b_t separator
        assert result.index("######## a_t ########") < result.index("######## b_t ########")

    def test_tabs_indent(self):
        lines = ["allow a_t b_t:file read;"]
        result = spm.make_it_shine(lines, None, "TITLE", ">>")
        assert ">>" in result

    def test_sorts_lines(self):
        lines = ["allow z_t a_t:file read;", "allow a_t z_t:file write;"]
        result = spm.make_it_shine(lines, None, "TITLE")
        idx_a = result.index("allow a_t")
        idx_z = result.index("allow z_t")
        assert idx_a < idx_z

    def test_non_matching_line_with_pattern_logs_error(self, capsys):
        """When a pattern is given but a line does not match, make_it_shine skips it with an error."""
        lines = ["not a valid allow rule;"]
        result = spm.make_it_shine(lines, spm.REGEX_ALLOW_TEMPLATE, "ALLOW RULES")
        err = capsys.readouterr().err
        assert "error parsing rule" in err
        # The non-matching line should not appear in the output
        assert "not a valid allow rule" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRegexPatterns:
    # ── REGEX_ALLOW_TEMPLATE ────────────────────────────────────────────────
    def test_allow_multi_perm(self):
        m = spm.REGEX_ALLOW_TEMPLATE.match("allow myapp_t tmp_t:file { read write };")
        assert m is not None
        assert m.group(1) == "myapp_t"
        assert m.group(2) == "tmp_t"
        assert m.group(3) == "file"
        assert "read" in m.group(4)
        assert "write" in m.group(4)

    def test_allow_single_perm(self):
        m = spm.REGEX_ALLOW_TEMPLATE.match("allow init_t myapp_t:process transition;")
        assert m is not None
        assert m.group(1) == "init_t"
        assert m.group(5) == "transition"

    def test_allow_no_match(self):
        m = spm.REGEX_ALLOW_TEMPLATE.match("deny a_t b_t:file read;")
        assert m is None

    @pytest.mark.parametrize("line", [
        "allow a_t b_t:file read;",
        "allow a_t b_t:file { read };",
        "allow a_t b_t:file { read write };",
        "allow a_t b_t:file { read write getattr };",
        "allow init_t myapp_exec_t:process transition;",
    ])
    def test_allow_matches(self, line):
        """All valid allow rule forms should match REGEX_ALLOW_TEMPLATE."""
        assert spm.REGEX_ALLOW_TEMPLATE.match(line) is not None

    @pytest.mark.parametrize("line", [
        "deny a_t b_t:file read;",
        "allow a_t b_t file read;",
        "ALLOW a_t b_t:file read;",
        "",
    ])
    def test_allow_rejects(self, line):
        """Invalid allow rule forms should not match REGEX_ALLOW_TEMPLATE."""
        assert spm.REGEX_ALLOW_TEMPLATE.match(line) is None

    # ── REGEX_ROLE_TEMPLATE ─────────────────────────────────────────────────
    def test_role_multi(self):
        m = spm.REGEX_ROLE_TEMPLATE.match("role system_r types { myapp_t helper_t };")
        assert m is not None
        assert m.group(1) == "system_r"
        assert "myapp_t" in m.group(2)
        assert "helper_t" in m.group(2)

    def test_role_single(self):
        m = spm.REGEX_ROLE_TEMPLATE.match("role unconfined_r types myapp_t;")
        assert m is not None
        assert m.group(3) == "myapp_t"

    @pytest.mark.parametrize("line", [
        "role system_r types myapp_t;",
        "role system_r types { myapp_t };",
        "role system_r types { myapp_t helper_t };",
    ])
    def test_role_matches(self, line):
        assert spm.REGEX_ROLE_TEMPLATE.match(line) is not None

    @pytest.mark.parametrize("line", [
        "role system_r;",
        "ROLE system_r types myapp_t;",
        "",
    ])
    def test_role_rejects(self, line):
        assert spm.REGEX_ROLE_TEMPLATE.match(line) is None

    # ── REGEX_TYPE_TRANSITION_TEMPLATE ──────────────────────────────────────
    def test_type_transition(self):
        m = spm.REGEX_TYPE_TRANSITION_TEMPLATE.match(
            "type_transition init_t myapp_exec_t:process myapp_t;"
        )
        assert m is not None
        assert m.group(1) == "init_t"
        assert m.group(2) == "myapp_exec_t"
        assert m.group(3) == "process"
        assert m.group(4) == "myapp_t"

    def test_type_transition_trailing_space(self):
        m = spm.REGEX_TYPE_TRANSITION_TEMPLATE.match(
            "type_transition init_t exec_t:process app_t ;"
        )
        assert m is not None

    @pytest.mark.parametrize("line", [
        "type_transition init_t exec_t:process app_t;",
        "type_transition init_t exec_t:process app_t ;",
        "type_transition unconfined_t myapp_exec_t:process myapp_t;",
    ])
    def test_type_transition_matches(self, line):
        assert spm.REGEX_TYPE_TRANSITION_TEMPLATE.match(line) is not None

    @pytest.mark.parametrize("line", [
        "type_transition init_t;",
        "TYPE_TRANSITION init_t exec_t:process app_t;",
        "",
    ])
    def test_type_transition_rejects(self, line):
        assert spm.REGEX_TYPE_TRANSITION_TEMPLATE.match(line) is None

    # ── REGEX_TYPE_DEFINITION_TEMPLATE ──────────────────────────────────────
    def test_type_definition_single(self):
        m = spm.REGEX_TYPE_DEFINITION_TEMPLATE.match("type myapp_t;")
        assert m is not None
        assert m.group(1) == "myapp_t"

    def test_type_definition_multi(self):
        m = spm.REGEX_TYPE_DEFINITION_TEMPLATE.match("type myapp_t, myapp_exec_t;")
        assert m is not None
        assert "myapp_t" in m.group(1)
        assert "myapp_exec_t" in m.group(1)

    @pytest.mark.parametrize("line", [
        "type myapp_t;",
        "type myapp_t, myapp_exec_t;",
        "type a_t, b_t, c_t;",
    ])
    def test_type_definition_matches(self, line):
        assert spm.REGEX_TYPE_DEFINITION_TEMPLATE.match(line) is not None

    # ── REGEX_TYPEATTRIBUTE_TEMPLATE ────────────────────────────────────────
    def test_typeattribute_single(self):
        m = spm.REGEX_TYPEATTRIBUTE_TEMPLATE.match("typeattribute myapp_t file_type;")
        assert m is not None
        assert m.group(1) == "myapp_t"
        assert m.group(2) == "file_type"

    def test_typeattribute_multi(self):
        m = spm.REGEX_TYPEATTRIBUTE_TEMPLATE.match(
            "typeattribute myapp_t file_type, domain;"
        )
        assert m is not None
        assert "file_type" in m.group(2)
        assert "domain" in m.group(2)

    @pytest.mark.parametrize("line", [
        "typeattribute myapp_t file_type;",
        "typeattribute myapp_t file_type, domain;",
        "typeattribute myapp_t file_type, domain, netlabel_peer_type;",
    ])
    def test_typeattribute_matches(self, line):
        assert spm.REGEX_TYPEATTRIBUTE_TEMPLATE.match(line) is not None

    # ── REGEX_ROLE_TRANSITION_TEMPLATE ──────────────────────────────────────
    def test_role_transition(self):
        m = spm.REGEX_ROLE_TRANSITION_TEMPLATE.match(
            "role_transition unconfined_r myapp_exec_t system_r;"
        )
        assert m is not None
        assert m.group(1) == "unconfined_r"
        assert m.group(2) == "myapp_exec_t"
        assert m.group(3) == "system_r"

    @pytest.mark.parametrize("line", [
        "role_transition unconfined_r myapp_exec_t system_r;",
        "role_transition staff_r helper_exec_t system_r;",
    ])
    def test_role_transition_matches(self, line):
        assert spm.REGEX_ROLE_TRANSITION_TEMPLATE.match(line) is not None

    @pytest.mark.parametrize("line", [
        "role_transition unconfined_r;",
        "ROLE_TRANSITION unconfined_r myapp_exec_t system_r;",
        "",
    ])
    def test_role_transition_rejects(self, line):
        assert spm.REGEX_ROLE_TRANSITION_TEMPLATE.match(line) is None


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — merger()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMerger:
    def test_minimal_allow(self):
        result = spm.merger(_lines(MINIMAL_ALLOW))
        assert "module myapp 1.0;" in result
        assert "allow myapp_t tmp_t:file { read };" in result
        assert "require {" in result
        assert "type myapp_t, tmp_t;" in result
        assert "class file { read };" in result

    def test_multi_perm_allow(self):
        result = spm.merger(_lines(MULTI_PERM_ALLOW))
        assert "allow myapp_t tmp_t:file { getattr read write };" in result

    def test_single_perm_allow(self):
        result = spm.merger(_lines(SINGLE_PERM_ALLOW))
        assert "allow init_t myapp_t:process { transition };" in result

    def test_role_rules(self):
        result = spm.merger(_lines(ROLE_POLICY))
        assert "ROLE RULES" in result
        assert "role system_r types { helper_t myapp_t };" in result
        assert "role unconfined_r types { myapp_t };" in result

    def test_type_transition(self):
        result = spm.merger(_lines(TYPE_TRANSITION_POLICY))
        assert "type_transition init_t myapp_exec_t:process myapp_t;" in result
        assert "type_transition unconfined_t myapp_exec_t:process myapp_t;" in result

    def test_type_definitions(self):
        result = spm.merger(_lines(TYPE_DEFINITION_POLICY))
        assert "type myapp_t;" in result
        assert "type myapp_exec_t;" in result
        assert "type myapp_file_t;" in result

    def test_typeattribute(self):
        result = spm.merger(_lines(TYPEATTRIBUTE_POLICY))
        assert "typeattribute myapp_exec_t file_type;" in result
        assert "typeattribute myapp_file_t domain, file_type;" in result

    def test_role_transition(self):
        result = spm.merger(_lines(ROLE_TRANSITION_POLICY))
        assert "role_transition unconfined_r myapp_exec_t system_r;" in result

    def test_full_policy_all_sections(self):
        result = spm.merger(_lines(FULL_POLICY))
        # Module line
        assert "module myapp 1.0;" in result
        # Type definitions
        assert "type myapp_t;" in result
        assert "type myapp_exec_t;" in result
        assert "type myapp_file_t;" in result
        # Require block
        assert "require {" in result
        assert "role system_r" in result
        assert "attribute file_type;" in result
        # All section headers
        assert "ROLE RULES" in result
        assert "TYPE TRANSITION RULES" in result
        assert "ROLE TRANSITION RULES" in result
        assert "TYPE ATTRIBUTES" in result
        assert "ALLOW RULES" in result

    def test_require_block_stripped_and_regenerated(self):
        result = spm.merger(_lines(REQUIRE_BLOCK_POLICY))
        # The original require block is gone; a new one is generated
        assert "require {" in result
        assert "type init_t" in result
        # Note: myapp_t is defined in the module but the tool also lists it in
        # required_types (gathered from allow rules); both sets are independent.
        # Types from rules should be in the new require block
        lines = result.split("\n")
        in_require = False
        req_types = ""
        for line in lines:
            if "require {" in line:
                in_require = True
            if in_require:
                req_types += line
            if in_require and "}" in line and "require" not in line:
                break
        assert "init_t" in req_types
        assert "tmp_t" in req_types

    def test_module_title_extracted(self):
        lines = _lines("module custom 2.5;\nallow a_t b_t:file read;\n")
        result = spm.merger(lines)
        assert "module custom 2.5;" in result

    def test_self_excluded_from_require(self):
        lines = _lines("module myapp 1.0;\nallow myapp_t self:file read;\n")
        result = spm.merger(lines)
        # 'self' should not appear in the require block's type list
        req_start = result.index("require {")
        req_end = result.index("}", req_start)
        require_block = result[req_start:req_end]
        assert "self" not in require_block

    def test_empty_input(self):
        result = spm.merger([])
        assert "module myapp 1.0;" in result
        # No sections should appear
        assert "ALLOW RULES" not in result
        assert "require {" not in result

    def test_defined_type_also_in_require_block(self):
        """A type that is both defined and referenced in allow rules appears in both sections."""
        lines = _lines(textwrap.dedent("""\
            module myapp 1.0;
            type myapp_t;
            allow myapp_t tmp_t:file read;
        """))
        result = spm.merger(lines)
        # It should be in the type definition section
        assert "type myapp_t;" in result
        # It should also appear in the require block (current behaviour: not filtered out)
        req_start = result.index("require {")
        req_end = result.index("}", req_start)
        require_block = result[req_start:req_end]
        assert "myapp_t" in require_block

    def test_comments_only_input(self):
        """Input with only comments produces a minimal valid module, no sections."""
        result = spm.merger(_lines("# just comments\n# nothing else\n"))
        assert "module myapp 1.0;" in result
        assert "ALLOW RULES" not in result
        assert "ROLE RULES" not in result
        assert "require {" not in result

    def test_only_type_definitions_no_rules(self):
        """Input with only module + type definitions emits types but no require/allow sections."""
        result = spm.merger(_lines("module myapp 1.0;\ntype foo_t;\ntype bar_t;\n"))
        assert "type bar_t;" in result
        assert "type foo_t;" in result
        assert "require {" not in result
        assert "ALLOW RULES" not in result

    def test_output_section_ordering(self):
        """Verify that output sections appear in the canonical order."""
        result = spm.merger(_lines(FULL_POLICY))
        # Gather positions of key landmarks
        pos_module = result.index("module myapp 1.0;")
        pos_type_def = result.index("type myapp_exec_t;")
        pos_require = result.index("require {")
        pos_role = result.index("ROLE RULES")
        pos_tt = result.index("TYPE TRANSITION RULES")
        pos_rt = result.index("ROLE TRANSITION RULES")
        pos_ta = result.index("TYPE ATTRIBUTES")
        pos_allow = result.index("ALLOW RULES")

        assert pos_module < pos_type_def < pos_require
        assert pos_require < pos_role < pos_tt < pos_rt < pos_ta < pos_allow

    def test_no_module_line_uses_default(self):
        """When input lacks a module statement, the default 'myapp 1.0' is used."""
        result = spm.merger(_lines("allow a_t b_t:file read;\n"))
        assert "module myapp 1.0;" in result

    def test_cross_class_permissions_independent(self):
        """Permissions for different classes on the same subject-target are independent."""
        lines = _lines(textwrap.dedent("""\
            module myapp 1.0;
            allow a_t b_t:file read;
            allow a_t b_t:dir search;
        """))
        result = spm.merger(lines)
        assert "allow a_t b_t:dir { search };" in result
        assert "allow a_t b_t:file { read };" in result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-FILE MERGE TESTS — deduplication
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMultiFileMerge:
    def test_overlapping_allow_rules_merged(self):
        combined = _lines(POLICY_A) + _lines(POLICY_B)
        result = spm.merger(combined)
        # init_t:file should have read + execute merged
        assert re.search(r"allow myapp_t init_t:file \{.*read.*\}", result)
        assert re.search(r"allow myapp_t init_t:file \{.*execute.*\}", result)
        # Both unique rules should be present
        assert "allow myapp_t tmp_t:file" in result
        assert "allow myapp_t bin_t:file" in result

    def test_overlapping_type_definitions_deduplicated(self):
        combined = _lines(POLICY_A) + _lines(POLICY_B)
        result = spm.merger(combined)
        # myapp_t should appear exactly once as a type definition
        type_def_lines = [l for l in result.split("\n")
                          if l.strip().startswith("type ") and l.strip().endswith(";")
                          and "require" not in l and "type_transition" not in l
                          and "typeattribute" not in l]
        type_names = []
        for l in type_def_lines:
            # "type myapp_t;" -> "myapp_t"
            name = l.strip()[5:-1].strip()
            type_names.append(name)
        assert type_names.count("myapp_t") == 1

    def test_module_title_last_wins(self):
        a = _lines("module alpha 1.0;\nallow a_t b_t:file read;\n")
        b = _lines("module beta 2.0;\nallow c_t d_t:file write;\n")
        result = spm.merger(a + b)
        # The module line that sort_rules_by_type processes last wins
        assert "module beta 2.0;" in result

    def test_require_block_merged_from_both(self):
        combined = _lines(POLICY_A) + _lines(POLICY_B)
        result = spm.merger(combined)
        # All types from both files should be in the require block
        req_start = result.index("require {")
        req_end = result.index("}", req_start)
        require_block = result[req_start:req_end]
        for t in ["init_t", "tmp_t", "bin_t"]:
            assert t in require_block

    def test_require_classes_merged(self):
        combined = _lines(POLICY_A) + _lines(POLICY_B)
        result = spm.merger(combined)
        # file class should have read + write + execute
        m = re.search(r"class file \{ (.*?) \}", result)
        assert m is not None
        perms = set(m.group(1).split())
        assert {"read", "write", "execute"}.issubset(perms)


# ═══════════════════════════════════════════════════════════════════════════════
# IDEMPOTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIdempotency:
    def test_merger_output_is_idempotent(self):
        """merger(merger(input)) should produce the same output as merger(input)."""
        first = spm.merger(_lines(FULL_POLICY))
        second = spm.merger(_lines(first))
        assert first == second

    def test_minimal_idempotent(self):
        first = spm.merger(_lines(MINIMAL_ALLOW))
        second = spm.merger(_lines(first))
        assert first == second

    def test_role_policy_idempotent(self):
        first = spm.merger(_lines(ROLE_POLICY))
        second = spm.merger(_lines(first))
        assert first == second

    def test_multi_file_idempotent(self):
        combined = _lines(POLICY_A) + _lines(POLICY_B)
        first = spm.merger(combined)
        second = spm.merger(_lines(first))
        assert first == second


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCounters:
    def test_allow_counter(self):
        spm.merger(_lines(MULTI_PERM_ALLOW))
        assert spm.allow_counter == 3  # read, write, getattr

    def test_role_counter(self):
        spm.merger(_lines(ROLE_POLICY))
        # system_r has {myapp_t, helper_t} = 2, unconfined_r has {myapp_t} = 1
        assert spm.role_counter == 3

    def test_type_transition_counter(self):
        spm.merger(_lines(TYPE_TRANSITION_POLICY))
        assert spm.type_transition_counter == 2

    def test_type_definition_counter(self):
        spm.merger(_lines(TYPE_DEFINITION_POLICY))
        assert spm.type_definition_counter == 3

    def test_typeattribute_counter(self):
        spm.merger(_lines(TYPEATTRIBUTE_POLICY))
        # myapp_exec_t: file_type (1), myapp_file_t: file_type + domain (2)
        assert spm.typeattribute_counter == 3

    def test_role_transition_counter(self):
        spm.merger(_lines(ROLE_TRANSITION_POLICY))
        assert spm.role_transition_counter == 1

    def test_counters_reset_between_calls(self):
        spm.merger(_lines(MULTI_PERM_ALLOW))
        assert spm.allow_counter == 3
        spm.merger(_lines(MINIMAL_ALLOW))
        assert spm.allow_counter == 1


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC POLICY TESTS — synthetic MYAPP-like policies
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRealisticPolicies:
    def test_myapp_allow_policy_parses(self):
        result = spm.merger(_lines(MYAPP_ALLOW_POLICY))
        assert "module myapp" in result
        assert "allow myapp_t" in result
        assert "require {" in result

    def test_myapp_bootstrap_parses(self):
        result = spm.merger(_lines(MYAPP_BOOTSTRAP_POLICY))
        assert "module myapp" in result
        assert "myapp_t" in result

    def test_merge_both_policies(self):
        combined = _lines(MYAPP_ALLOW_POLICY) + _lines(MYAPP_BOOTSTRAP_POLICY)
        result = spm.merger(combined)
        assert "module myapp" in result
        assert "allow myapp_t" in result
        assert spm.allow_counter > 0
        # Should have at least as many rules as bootstrap alone
        bootstrap_only = spm.merger(_lines(MYAPP_BOOTSTRAP_POLICY))
        bootstrap_allow = spm.allow_counter
        spm.merger(combined)
        assert spm.allow_counter >= bootstrap_allow

    def test_merged_is_idempotent(self):
        combined = _lines(MYAPP_ALLOW_POLICY) + _lines(MYAPP_BOOTSTRAP_POLICY)
        first = spm.merger(combined)
        second = spm.merger(_lines(first))
        assert first == second

    def test_no_errors_on_stderr(self, capsys):
        combined = _lines(MYAPP_ALLOW_POLICY) + _lines(MYAPP_BOOTSTRAP_POLICY)
        spm.merger(combined)
        err = capsys.readouterr().err
        assert "-!- error" not in err


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TESTS — argparse and file I/O
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
class TestCLI:
    def test_stdout_output(self, tmp_path):
        policy_path = tmp_path / "input.te"
        policy_path.write_text(MINIMAL_ALLOW)
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(policy_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "module myapp 1.0;" in result.stdout
        assert "allow myapp_t tmp_t:file" in result.stdout
        # Counter summary on stderr
        assert "allow rule(s)" in result.stderr

    def test_dest_flag(self, tmp_path):
        policy_path = tmp_path / "input.te"
        policy_path.write_text(MINIMAL_ALLOW)
        dest_path = tmp_path / "output.te"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(policy_path),
             "--dest", str(dest_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert dest_path.exists()
        content = dest_path.read_text()
        assert "module myapp 1.0;" in content

    def test_multiple_files(self, tmp_path):
        a_path = tmp_path / "a.te"
        b_path = tmp_path / "b.te"
        a_path.write_text(POLICY_A)
        b_path.write_text(POLICY_B)
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(a_path), str(b_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "allow myapp_t bin_t:file" in result.stdout
        assert "allow myapp_t tmp_t:file" in result.stdout

    def test_verbose_flag(self, tmp_path):
        policy_path = tmp_path / "input.te"
        policy_path.write_text(MINIMAL_ALLOW)
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(policy_path), "-v"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # -v enables show_info; the module title info line should appear
        assert "module title is now" in result.stderr
        # Summary is always emitted regardless of -v
        assert "allow rule(s)" in result.stderr

    def test_debug_flag(self, tmp_path):
        policy_path = tmp_path / "input.te"
        # Include trailing newlines so the parser encounters empty lines
        policy_path.write_text(MINIMAL_ALLOW + "\n")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(policy_path), "-d"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # -d enables show_debug; empty lines produce "info: got null line" on stderr
        assert "got null line" in result.stderr

    def test_missing_files_flag(self):
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_nonexistent_file(self, tmp_path):
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--files", str(tmp_path / "missing.te")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "No such file" in result.stderr or "FileNotFoundError" in result.stderr
