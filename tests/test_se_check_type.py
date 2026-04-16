#!/usr/bin/env python3
"""
Test suite for se_check_type.

All tests run without seinfo / semodule / checkmodule — the host-level
SELinux tooling is fully mocked so the suite is portable.

Test structure (bottom-up)
──────────────────────────
  1. Regex patterns         - compiled regex validation
  2. Field/rule extraction  - _extract_names_from_field, extract_types_from_rules
  3. Document parsing       - _get_require_block_indices, _rewrite_require_types
  4. CLI & seinfo parsing   - parse_args, parse_seinfo_output, parse_policy
  5. Coherency checks       - check_coherency
  6. Remove/create logic    - process_missing_type_remove, process_missing_type_create
  7. Integration (main)     - report, remove, create modes with mocked seinfo
  8. Error paths            - seinfo failures, undeclared types, alias errors
  9. Edge cases             - malformed input, empty blocks, debug output
"""

import os
import subprocess
import tempfile
import textwrap
from unittest import mock

import pytest  # pyright: ignore[reportMissingImports]

from conftest import se_check_type as sct, _PROJECT_ROOT

_SCRIPT_PATH = os.path.join(_PROJECT_ROOT, "se_check_type")


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_debug():
    """Ensure the module-level _debug flag is always reset between tests."""
    yield
    sct._debug = False


@pytest.fixture
def write_policy(tmp_path):
    """Return a helper that writes a policy file and returns its path."""
    def _write(content, name="myapp.te"):
        p = tmp_path / name
        p.write_text(content)
        return str(p)
    return _write


@pytest.fixture
def mock_seinfo():
    """Return a context-manager factory that patches seinfo subprocess calls."""
    real_run = subprocess.run
    def _factory(stdout=FAKE_SEINFO_OUTPUT, returncode=0):
        def side_effect(cmd, *args, **kwargs):
            if cmd[0] == "seinfo":
                return mock.Mock(returncode=returncode, stdout=stdout, stderr="")
            return real_run(cmd, *args, **kwargs)
        return mock.patch("se_check_type.subprocess.run", side_effect=side_effect)
    return _factory


# ── sample policy documents ────────────────────────────────────────────────
SIMPLE_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type httpd_t, tmp_t, var_log_t;
    }

    allow myapp_t httpd_t:file read;
    allow myapp_t tmp_t:dir write;
    allow myapp_t var_log_t:file append;
""")

MULTI_TYPE_LINES_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type httpd_t;
        type tmp_t;
        type var_log_t;
    }

    allow myapp_t httpd_t:file read;
    allow myapp_t tmp_t:dir write;
    allow myapp_t var_log_t:file append;
""")

# Policy with a type used in rules but not declared in require
INCOHERENT_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type httpd_t, tmp_t;
    }

    allow myapp_t httpd_t:file read;
    allow myapp_t tmp_t:dir write;
    allow myapp_t var_log_t:file append;
""")

# Policy with declared but unused type
UNUSED_TYPE_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type httpd_t, tmp_t, var_log_t, unused_t;
    }

    allow myapp_t httpd_t:file read;
    allow myapp_t tmp_t:dir write;
    allow myapp_t var_log_t:file append;
""")

# Policy with set-based rules (for remove testing)
SET_POLICY = textwrap.dedent("""\
    module myapp 1.0;

    type myapp_t;

    require {
        type httpd_t, missing_t, other_t;
    }

    allow myapp_t { httpd_t missing_t other_t }:file read;
    allow myapp_t missing_t:dir write;
""")

# Minimal fake seinfo output
FAKE_SEINFO_OUTPUT = textwrap.dedent("""\
    type httpd_t;
    type tmp_t;
    type var_log_t;
    type container_t, alias {old_container_t legacy_t},;
    type sshd_t;
    type init_t;
""")

FAKE_SEINFO_SINGLE_ALIAS = textwrap.dedent("""\
    type httpd_t;
    type tmp_t;
    type var_log_t;
    type container_t, alias old_container_t,;
""")


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX TESTS — verify the compiled patterns work correctly
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRegexPatterns:
    """Compiled regex patterns used throughout the module."""
    def test_regex_get_first_type(self):
        text = "type myapp_t;\ntype other_t;"
        matches = sct.REGEX_GET_FIRST_TYPE.findall(text)
        assert "myapp_t" in matches
        assert "other_t" in matches

    def test_regex_types_line(self):
        text = "type httpd_t, tmp_t;"
        matches = sct.REGEX_TYPES_LINE.findall(text)
        assert "httpd_t, tmp_t" in matches

    def test_regex_seinfo_aliases_multi(self):
        text = "container_t, alias {old_container_t legacy_t},"
        match = sct.REGEX_SEINFO_ALIASES.search(text)
        assert match is not None
        assert "old_container_t" in match.group(1)

    def test_regex_seinfo_aliases_single(self):
        text = "container_t, alias old_container_t,"
        match = sct.REGEX_SEINFO_ALIASES.search(text)
        assert match is not None
        assert match.group(1) == "old_container_t"

    def test_regex_av_rule(self):
        text = "allow myapp_t httpd_t:file read;"
        match = sct.REGEX_AV_RULE.search(text)
        assert match is not None
        assert match.group(1) == "myapp_t"
        assert "httpd_t" in match.group(2)

    def test_regex_av_rule_set_source(self):
        text = "allow { src1_t src2_t } dest_t:file read;"
        match = sct.REGEX_AV_RULE.search(text)
        assert match is not None
        assert "src1_t" in match.group(1)
        assert "src2_t" in match.group(1)

    def test_regex_type_rule(self):
        text = "type_transition myapp_t tmp_t:file myapp_tmp_t;"
        match = sct.REGEX_TYPE_RULE.search(text)
        assert match is not None
        assert match.group(1) == "myapp_t"
        assert "tmp_t" in match.group(2)
        assert match.group(3) == "myapp_tmp_t;"

    def test_regex_role_types(self):
        text = "role myrole types { myapp_t other_t };"
        match = sct.REGEX_ROLE_TYPES.search(text)
        assert match is not None
        assert "myapp_t" in match.group(1)
        assert "other_t" in match.group(1)

    def test_regex_role_transition(self):
        text = "role_transition sysadm_r httpd_t system_r;"
        match = sct.REGEX_ROLE_TRANSITION.search(text)
        assert match is not None
        assert match.group(1) == "httpd_t"

    def test_regex_typeattribute(self):
        text = "typeattribute myapp_t domain;"
        match = sct.REGEX_TYPEATTRIBUTE.search(text)
        assert match is not None
        assert match.group(1) == "myapp_t"

    def test_regex_valid_type_name(self):
        assert sct.REGEX_VALID_TYPE_NAME.match("httpd_t")
        assert sct.REGEX_VALID_TYPE_NAME.match("my_custom_type")
        assert not sct.REGEX_VALID_TYPE_NAME.match("../evil")
        assert not sct.REGEX_VALID_TYPE_NAME.match("some type")

    def test_regex_set_content(self):
        text = "allow a { b c d }:file read;"
        match = sct.REGEX_SET_CONTENT.search(text)
        assert match is not None
        assert "b c d" in match.group(1)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _extract_names_from_field
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExtractNamesFromField:
    """_extract_names_from_field(): parse type names from rule fields."""
    def test_simple_name(self):
        assert sct._extract_names_from_field("httpd_t") == {"httpd_t"}

    def test_name_with_class_suffix(self):
        assert sct._extract_names_from_field("httpd_t:file") == {"httpd_t"}

    def test_set(self):
        result = sct._extract_names_from_field("{ type_a type_b type_c }")
        assert result == {"type_a", "type_b", "type_c"}

    def test_negation(self):
        assert sct._extract_names_from_field("~httpd_t") == {"httpd_t"}

    def test_set_with_exclusion(self):
        result = sct._extract_names_from_field("{ type_a -type_b }")
        assert result == {"type_a", "type_b"}

    def test_empty_string(self):
        assert sct._extract_names_from_field("") == set()

    def test_trailing_semicolon(self):
        assert sct._extract_names_from_field("httpd_t;") == {"httpd_t"}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — extract_types_from_rules
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExtractTypesFromRules:
    """extract_types_from_rules(): collect all type names from policy rule body."""
    def test_allow_rule(self):
        body = "allow myapp_t httpd_t:file read;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "httpd_t" in types

    def test_dontaudit_rule(self):
        body = "dontaudit myapp_t tmp_t:dir search;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "tmp_t" in types

    def test_neverallow_rule(self):
        body = "neverallow myapp_t secret_t:file write;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "secret_t" in types

    def test_auditallow_rule(self):
        body = "auditallow myapp_t audit_t:file read;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "audit_t" in types

    def test_type_transition(self):
        body = "type_transition myapp_t tmp_t:file myapp_tmp_t;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "tmp_t" in types
        assert "myapp_tmp_t" in types

    def test_type_change(self):
        body = "type_change myapp_t var_t:file myapp_var_t;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "var_t" in types
        assert "myapp_var_t" in types

    def test_type_member(self):
        body = "type_member myapp_t container_t:file myapp_file_t;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "container_t" in types
        assert "myapp_file_t" in types

    def test_role_types(self):
        body = "role myrole types { myapp_t other_t };"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "other_t" in types

    def test_role_transition(self):
        body = "role_transition sysadm_r httpd_t system_r;"
        types = sct.extract_types_from_rules(body)
        assert "httpd_t" in types

    def test_typeattribute(self):
        body = "typeattribute myapp_t domain;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types

    def test_self_excluded(self):
        body = "allow myapp_t self:process fork;"
        types = sct.extract_types_from_rules(body)
        assert "myapp_t" in types
        assert "self" not in types

    def test_comments_stripped(self):
        body = "# allow fake_t fake2_t:file read;\nallow myapp_t httpd_t:file read;"
        types = sct.extract_types_from_rules(body)
        assert "fake_t" not in types
        assert "myapp_t" in types

    def test_set_source(self):
        body = "allow { src1_t src2_t } dest_t:file read;"
        types = sct.extract_types_from_rules(body)
        assert "src1_t" in types
        assert "src2_t" in types
        assert "dest_t" in types

    def test_empty_body(self):
        types = sct.extract_types_from_rules("")
        assert len(types) == 0

    def test_multiple_rules(self):
        body = textwrap.dedent("""\
            allow a_t b_t:file read;
            dontaudit c_t d_t:dir search;
            type_transition e_t f_t:file g_t;
            typeattribute h_t domain;
        """)
        types = sct.extract_types_from_rules(body)
        for t in ["a_t", "b_t", "c_t", "d_t", "e_t", "f_t", "g_t", "h_t"]:
            assert t in types


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _get_require_block_indices
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetRequireBlockIndices:
    """_get_require_block_indices(): locate require { ... } in policy document."""
    def test_simple_block(self):
        start, end = sct._get_require_block_indices(SIMPLE_POLICY)
        block = SIMPLE_POLICY[start:end + 1]
        assert block.startswith("require {")
        assert block.endswith("}")
        assert "httpd_t" in block

    def test_nested_braces(self):
        doc = "preamble require { type a_t; role r types { a_t }; } trailing"
        start, end = sct._get_require_block_indices(doc)
        block = doc[start:end + 1]
        assert block == "require { type a_t; role r types { a_t }; }"

    def test_missing_require_raises(self):
        with pytest.raises(ValueError, match="not found"):
            sct._get_require_block_indices("module foo 1.0; type foo_t;")

    def test_unclosed_require_raises(self):
        with pytest.raises(ValueError, match="never closed"):
            sct._get_require_block_indices("require { type foo_t;")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _rewrite_require_types
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRewriteRequireTypes:
    """_rewrite_require_types(): consolidate type declarations in require block."""
    def test_consolidates_single_line(self):
        result = sct._rewrite_require_types(SIMPLE_POLICY, ["alpha_t", "beta_t"], "    ")
        assert "type alpha_t, beta_t;" in result
        # original types should be gone from require block
        start, end = sct._get_require_block_indices(result)
        require_block = result[start:end + 1]
        assert "httpd_t" not in require_block

    def test_consolidates_multi_line(self):
        result = sct._rewrite_require_types(MULTI_TYPE_LINES_POLICY, ["x_t", "y_t", "z_t"], "    ")
        assert "type x_t, y_t, z_t;" in result
        # all old type lines removed
        assert "type httpd_t;" not in result
        assert "type tmp_t;" not in result
        assert "type var_log_t;" not in result

    def test_preserves_content_outside_require(self):
        result = sct._rewrite_require_types(SIMPLE_POLICY, ["httpd_t"], "    ")
        assert "module myapp 1.0;" in result
        assert "allow myapp_t httpd_t:file read;" in result

    def test_custom_indent(self):
        result = sct._rewrite_require_types(SIMPLE_POLICY, ["a_t"], "\t\t")
        assert "\t\ttype a_t;" in result


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — parse_args
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseArgs:
    """parse_args(): CLI argument parsing."""
    def test_basic(self):
        args = sct.parse_args(["--policy", "myapp.te"])
        assert args.policy == "myapp.te"
        assert args.d is False
        assert args.c is False
        assert args.r is False
        assert args.ignore == []

    def test_all_flags(self):
        args = sct.parse_args(["--policy", "x.te", "-d", "-r", "--ignore", "foo_t", "bar_t"])
        assert args.d is True
        assert args.r is True
        assert args.ignore == ["foo_t", "bar_t"]

    def test_missing_policy_exits(self):
        with pytest.raises(SystemExit):
            sct.parse_args([])

    def test_create_flag(self):
        args = sct.parse_args(["--policy", "x.te", "-c"])
        assert args.c is True
        assert args.r is False

    def test_create_with_ignore(self):
        args = sct.parse_args(["--policy", "x.te", "-c", "--ignore", "foo_t"])
        assert args.c is True
        assert args.ignore == ["foo_t"]

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — parse_seinfo_output
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseSeinfo:
    """parse_seinfo_output(): extract types and aliases from seinfo output."""
    def test_basic_types(self):
        existing, all_types, aliases = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        assert "httpd_t" in existing
        assert "tmp_t" in existing
        assert "var_log_t" in existing
        assert "sshd_t" in existing

    def test_multi_alias(self):
        existing, all_types, aliases = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        assert "old_container_t" in all_types
        assert "legacy_t" in all_types
        assert "old_container_t" not in existing  # aliases shouldn't be in existing_types
        assert aliases["old_container_t"] == "container_t"
        assert aliases["legacy_t"] == "container_t"

    def test_single_alias(self):
        existing, all_types, aliases = sct.parse_seinfo_output(FAKE_SEINFO_SINGLE_ALIAS)
        assert "old_container_t" in all_types
        assert aliases["old_container_t"] == "container_t"

    def test_empty_input(self):
        existing, all_types, aliases = sct.parse_seinfo_output("")
        assert len(existing) == 0
        assert len(all_types) == 0
        assert len(aliases) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — parse_policy
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParsePolicy:
    """parse_policy(): extract created types, required types, and rule body."""
    def test_simple(self):
        created, required, rb_end = sct.parse_policy(SIMPLE_POLICY)
        assert "myapp_t" in created
        assert required == {"httpd_t", "tmp_t", "var_log_t"}
        assert rb_end > 0

    def test_multi_type_lines(self):
        created, required, rb_end = sct.parse_policy(MULTI_TYPE_LINES_POLICY)
        assert required == {"httpd_t", "tmp_t", "var_log_t"}

    def test_rule_body_accessible(self):
        _, _, rb_end = sct.parse_policy(SIMPLE_POLICY)
        rule_body = SIMPLE_POLICY[rb_end + 1:]
        assert "allow myapp_t httpd_t:file read;" in rule_body


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — check_coherency
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCheckCoherency:
    """check_coherency(): detect unused/undeclared types in policy."""
    def test_coherent_policy(self):
        _, required, rb_end = sct.parse_policy(SIMPLE_POLICY)
        existing, all_types, _ = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        rule_body = SIMPLE_POLICY[rb_end + 1:]
        unused, undeclared = sct.check_coherency(
            required, rule_body, [], ["myapp_t"], all_types
        )
        assert len(unused) == 0
        assert len(undeclared) == 0

    def test_unused_type(self):
        _, required, rb_end = sct.parse_policy(UNUSED_TYPE_POLICY)
        existing, all_types, _ = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        rule_body = UNUSED_TYPE_POLICY[rb_end + 1:]
        unused, undeclared = sct.check_coherency(
            required, rule_body, [], ["myapp_t"], all_types
        )
        assert "unused_t" in unused

    def test_undeclared_type(self):
        _, required, rb_end = sct.parse_policy(INCOHERENT_POLICY)
        existing, all_types, _ = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        rule_body = INCOHERENT_POLICY[rb_end + 1:]
        unused, undeclared = sct.check_coherency(
            required, rule_body, [], ["myapp_t"], all_types
        )
        assert "var_log_t" in undeclared

    def test_ignored_types_skipped(self):
        _, required, rb_end = sct.parse_policy(UNUSED_TYPE_POLICY)
        existing, all_types, _ = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        rule_body = UNUSED_TYPE_POLICY[rb_end + 1:]
        unused, undeclared = sct.check_coherency(
            required, rule_body, ["unused_t"], ["myapp_t"], all_types
        )
        assert "unused_t" not in unused

    def test_used_but_non_existent_warning(self, capsys):
        """Types used in rules but not existing on host should trigger a warning."""
        rule_body = "allow myapp_t phantom_t:file read;"
        required = {"phantom_t"}
        existing_types_and_aliases = set()  # phantom_t is not on the host
        unused, undeclared = sct.check_coherency(
            required, rule_body, [], ["myapp_t"], existing_types_and_aliases
        )
        out = capsys.readouterr().out
        assert "phantom_t" in out
        assert "does not exist on the host" in out

    def test_created_types_excluded_from_coherency(self):
        """Types created by the policy itself should not appear as unused or undeclared."""
        rule_body = "allow myapp_t httpd_t:file read;"
        required = {"httpd_t"}
        existing, all_types, aliases = sct.parse_seinfo_output(FAKE_SEINFO_OUTPUT)
        unused, undeclared = sct.check_coherency(
            required, rule_body, [], ["myapp_t"], all_types
        )
        assert "myapp_t" not in unused
        assert "myapp_t" not in undeclared


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — process_missing_type_remove
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestProcessMissingTypeRemove:
    """process_missing_type_remove(): remove a missing type from rule lines."""
    def test_removes_standalone_line(self):
        lines = [
            "allow myapp_t missing_t:dir write;",
            "allow myapp_t httpd_t:file read;",
        ]
        result, rm_count = sct.process_missing_type_remove("missing_t", lines)
        assert rm_count == 1
        assert len(result) == 1
        assert "httpd_t" in result[0]

    def test_removes_from_set(self):
        lines = ["allow myapp_t { httpd_t missing_t other_t }:file read;"]
        result, rm_count = sct.process_missing_type_remove("missing_t", lines)
        assert rm_count == 0  # removed from set, line kept
        assert len(result) == 1
        assert "missing_t" not in result[0]
        assert "httpd_t" in result[0]
        assert "other_t" in result[0]

    def test_removes_last_from_set(self):
        lines = ["allow myapp_t { missing_t }:file read;"]
        result, rm_count = sct.process_missing_type_remove("missing_t", lines)
        assert rm_count == 1  # entire line removed (empty set)
        assert len(result) == 0

    def test_preserves_type_declaration_lines(self):
        """Lines starting with 'type ' in require block are preserved by this function."""
        lines = [
            "    type httpd_t, missing_t;",
            "allow myapp_t missing_t:dir write;",
        ]
        result, rm_count = sct.process_missing_type_remove("missing_t", lines)
        # The type declaration line is skipped (handled by _rewrite_require_types)
        # but the rule line is removed
        assert rm_count == 1

    def test_no_false_positive_substring(self):
        """'missing_t' should NOT match 'not_missing_t' (word boundary)."""
        lines = [
            "allow myapp_t not_missing_t:dir write;",
            "allow myapp_t missing_t:file read;",
        ]
        result, rm_count = sct.process_missing_type_remove("missing_t", lines)
        assert rm_count == 1
        assert any("not_missing_t" in l for l in result)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — process_missing_type_create
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestProcessMissingTypeCreate:
    """process_missing_type_create(): create a new SELinux type module."""
    def test_success(self, tmp_path):
        mock_run = mock.Mock(return_value=mock.Mock(returncode=0))
        result = sct.process_missing_type_create("new_type_t", str(tmp_path) + "/", run_command=mock_run)
        assert result is True
        assert mock_run.call_count == 3
        # check .te file was written
        te_file = tmp_path / "new_type_t.te"
        assert te_file.exists()
        content = te_file.read_text()
        assert "module new_type_t 1.0" in content

    def test_failure_on_checkmodule(self, tmp_path):
        mock_run = mock.Mock(return_value=mock.Mock(returncode=1))
        result = sct.process_missing_type_create("bad_t", str(tmp_path) + "/", run_command=mock_run)
        assert result is False
        assert mock_run.call_count == 1  # short-circuits on first failure

    def test_invalid_type_name(self, tmp_path, capsys):
        """Invalid type name: returns False, logs to stderr, and creates no files."""
        result = sct.process_missing_type_create("../evil", str(tmp_path) + "/")
        assert result is False
        captured = capsys.readouterr()
        assert "invalid characters" in captured.err
        assert not any(tmp_path.iterdir())

    def test_commands_called_in_order(self, tmp_path):
        mock_run = mock.Mock(return_value=mock.Mock(returncode=0))
        sct.process_missing_type_create("foo_t", str(tmp_path) + "/", run_command=mock_run)
        cmds = [call.args[0][0] for call in mock_run.call_args_list]
        assert cmds == ["checkmodule", "semodule_package", "semodule"]

    def test_failure_on_semodule_package(self, tmp_path):
        """semodule_package (2nd command) fails — only 2 calls should be made."""
        call_count = 0
        def mock_run(cmd, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            rc = 1 if cmd[0] == "semodule_package" else 0
            return mock.Mock(returncode=rc)
        result = sct.process_missing_type_create("mid_t", str(tmp_path) + "/", run_command=mock_run)
        assert result is False
        assert call_count == 2  # checkmodule OK, semodule_package FAIL, stops

    def test_failure_on_semodule(self, tmp_path):
        """semodule (3rd command) fails — all 3 calls should be made."""
        call_count = 0
        def mock_run(cmd, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            rc = 1 if cmd[0] == "semodule" else 0
            return mock.Mock(returncode=rc)
        result = sct.process_missing_type_create("late_t", str(tmp_path) + "/", run_command=mock_run)
        assert result is False
        assert call_count == 3  # all three called, last one fails


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — main() with mocked seinfo
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMainReportMode:
    """main() in default mode (no -r / -c): just reports missing types."""

    def test_all_types_exist(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with mock_seinfo():
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "summary:" in out
        assert "0 removed" in out
        assert "0 created" in out
        assert "missing" not in out

    def test_missing_type_reported(self, write_policy, mock_seinfo, capsys):
        # Add a type that won't be in fake seinfo output
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, nonexistent_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t nonexistent_t:file read;"
        )
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "missing 'nonexistent_t'" in out

    def test_alias_reported(self, write_policy, mock_seinfo, capsys):
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, old_container_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t old_container_t:file read;"
        )
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "alias" in out
        assert "container_t" in out

    def test_coherency_warnings(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(UNUSED_TYPE_POLICY)
        # unused_t won't be in seinfo, so it will be reported as missing too
        # Let's add it to seinfo
        seinfo_with_unused = FAKE_SEINFO_OUTPUT + "    type unused_t;\n"
        with mock_seinfo(stdout=seinfo_with_unused):
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "declared in require but not used" in out

    def test_ignore_flag(self, write_policy, mock_seinfo, capsys):
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, nonexistent_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t nonexistent_t:file read;"
        )
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path, "--ignore", "nonexistent_t"])
        out = capsys.readouterr().out
        assert "missing" not in out
        assert "ignoring the following types" in out


@pytest.mark.integration
class TestMainRemoveMode:
    """main() with -r: removes missing types from the policy."""

    def test_removes_missing_type(self, tmp_path, write_policy, mock_seinfo, capsys):
        policy = textwrap.dedent("""\
            module myapp 1.0;

            type myapp_t;

            require {
                type httpd_t, tmp_t, missing_t;
            }

            allow myapp_t httpd_t:file read;
            allow myapp_t tmp_t:dir write;
            allow myapp_t missing_t:file append;
        """)
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path, "-r"])

        out = capsys.readouterr().out
        assert "1 removed" in out

        # verify the file was rewritten
        new_content = (tmp_path / "myapp.te").read_text()
        assert "missing_t" not in new_content
        assert "httpd_t" in new_content
        assert "tmp_t" in new_content

    def test_removes_from_set_in_rule(self, tmp_path, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SET_POLICY)
        # 'other_t' is not in our fake seinfo, so it will also be "missing"
        seinfo = FAKE_SEINFO_OUTPUT + "    type other_t;\n"
        with mock_seinfo(stdout=seinfo):
            sct.main(["--policy", policy_path, "-r"])

        new_content = (tmp_path / "myapp.te").read_text()
        assert "missing_t" not in new_content
        # httpd_t and other_t should remain in the set
        assert "httpd_t" in new_content
        assert "other_t" in new_content

    def test_alias_renamed_in_remove_mode(self, tmp_path, write_policy, mock_seinfo, capsys):
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, old_container_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t old_container_t:file read;"
        )
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path, "-r"])

        out = capsys.readouterr().out
        assert "1 renamed" in out

        new_content = (tmp_path / "myapp.te").read_text()
        assert "old_container_t" not in new_content
        assert "container_t" in new_content
        # require block is re-sorted after rename
        assert "type container_t, httpd_t, tmp_t, var_log_t;" in new_content


@pytest.mark.integration
class TestMainCreateMode:
    """main() with -c: creates missing types (subprocess mocked)."""

    def test_rc_incompatible(self, write_policy, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with pytest.raises(SystemExit):
            sct.main(["--policy", policy_path, "-r", "-c"])
        err = capsys.readouterr().err
        assert "not compatible" in err

    def test_requires_root(self, write_policy, capsys):
        if os.getuid() == 0:
            pytest.skip("test must run as non-root")
        policy_path = write_policy(SIMPLE_POLICY)
        with pytest.raises(SystemExit):
            sct.main(["--policy", policy_path, "-c"])
        err = capsys.readouterr().err
        assert "root privileges" in err

    def test_alias_renamed_in_create_mode(self, tmp_path, write_policy, mock_seinfo, capsys):
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, old_container_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t old_container_t:file read;"
        )
        policy_path = write_policy(policy)
        with mock_seinfo(stdout=FAKE_SEINFO_SINGLE_ALIAS), mock.patch("os.getuid", return_value=0):
            sct.main(["--policy", policy_path, "-c"])

        out = capsys.readouterr().out
        assert "1 renamed" in out

        new_content = (tmp_path / "myapp.te").read_text()
        assert "old_container_t" not in new_content
        assert "container_t" in new_content
        assert "type container_t, httpd_t, tmp_t, var_log_t;" in new_content

    def test_creates_missing_type_end_to_end(self, tmp_path, write_policy, mock_seinfo, capsys):
        policy = textwrap.dedent("""\
            module myapp 1.0;

            type myapp_t;

            require {
                type httpd_t, brand_new_t;
            }

            allow myapp_t httpd_t:file read;
            allow myapp_t brand_new_t:dir write;
        """)
        policy_path = write_policy(policy)

        # The default run_command=subprocess.run is captured at definition time,
        # so we wrap process_missing_type_create to inject a mocked run_command.
        create_run = mock.Mock(return_value=mock.Mock(returncode=0))
        original_create = sct.process_missing_type_create
        def create_wrapper(type_name, dirname, run_command=None):
            return original_create(type_name, dirname, run_command=create_run)

        with mock_seinfo(), \
             mock.patch("se_check_type.process_missing_type_create", side_effect=create_wrapper), \
             mock.patch("os.getuid", return_value=0):
            sct.main(["--policy", policy_path, "-c"])

        out = capsys.readouterr().out
        assert "1 created" in out
        # .te file for the new type should exist (written by the real function)
        assert (tmp_path / "brand_new_t.te").exists()
        # verify checkmodule / semodule_package / semodule were called
        assert create_run.call_count == 3


@pytest.mark.integration
class TestMainSeInfoFailure:
    """main() when seinfo is unavailable or fails."""

    def test_seinfo_nonzero_exit(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with mock_seinfo(returncode=1):
            with pytest.raises(SystemExit):
                sct.main(["--policy", policy_path])
        err = capsys.readouterr().err
        assert "seinfo" in err

    def test_seinfo_no_types(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with mock_seinfo(stdout=""):
            with pytest.raises(SystemExit):
                sct.main(["--policy", policy_path])
        err = capsys.readouterr().err
        assert "no types" in err


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — main() undeclared type raises ValueError
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMainUndeclaredRaises:
    """main() raises ValueError when rules use undeclared types."""
    def test_undeclared_type_raises_value_error(self, write_policy, mock_seinfo, capsys):
        """Policy whose rules use a type not in require (and not created) should print
        a per-type warning and then raise ValueError."""
        policy_path = write_policy(INCOHERENT_POLICY)
        with mock_seinfo():
            with pytest.raises(ValueError, match="undeclared types"):
                sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "is used in rules but not declared in require" in out
        assert "var_log_t" in out


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — alias error path (alias not in aliases dict)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMainAliasErrorPath:
    """main() error path when alias target cannot be resolved."""
    def test_alias_not_found_in_dict(self, write_policy, mock_seinfo, capsys):
        """Type in existing_types_and_aliases but NOT in existing_types AND NOT in aliases dict."""
        # Craft seinfo output where a type is in all_types but not in existing_types
        # and not mapped in aliases. This is tricky because parse_seinfo_output derives
        # both from the same data. We mock parse_seinfo_output directly.
        policy = SIMPLE_POLICY.replace(
            "type httpd_t, tmp_t, var_log_t;",
            "type httpd_t, tmp_t, var_log_t, phantom_alias_t;"
        ).replace(
            "allow myapp_t var_log_t:file append;",
            "allow myapp_t var_log_t:file append;\nallow myapp_t phantom_alias_t:file read;"
        )
        policy_path = write_policy(policy)

        # existing_types won't contain phantom_alias_t, but existing_types_and_aliases will
        patched_existing = {"httpd_t", "tmp_t", "var_log_t"}
        patched_all = {"httpd_t", "tmp_t", "var_log_t", "phantom_alias_t"}
        patched_aliases = {}  # phantom_alias_t NOT mapped

        with mock_seinfo(), \
             mock.patch("se_check_type.parse_seinfo_output",
                        return_value=(patched_existing, patched_all, patched_aliases)):
            # Should not crash but print an error about alias not found
            # The coherency check at end will raise because phantom_alias_t is not
            # in existing_types_and_aliases for check_coherency (it is, but test
            # mainly verifies the error_print path)
            try:
                sct.main(["--policy", policy_path])
            except ValueError:
                pass  # expected from coherency check at end
        err = capsys.readouterr().err
        assert "alias" in err
        assert "phantom_alias_t" in err


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestEdgeCases:
    """Edge cases: malformed policies, empty blocks, debug output, etc."""
    def test_policy_no_require_block(self, write_policy, mock_seinfo, capsys):
        policy = "module myapp 1.0;\ntype myapp_t;\n"
        policy_path = write_policy(policy)
        with mock_seinfo():
            with pytest.raises(ValueError, match="not found"):
                sct.main(["--policy", policy_path])

    def test_empty_require_block(self, write_policy, mock_seinfo, capsys):
        policy = textwrap.dedent("""\
            module myapp 1.0;

            type myapp_t;

            require {
            }

            allow myapp_t httpd_t:file read;
        """)
        policy_path = write_policy(policy)
        with mock_seinfo():
            with pytest.raises(SystemExit):
                sct.main(["--policy", policy_path])
        err = capsys.readouterr().err
        assert "no types found" in err

    def test_debug_flag(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with mock_seinfo():
            sct.main(["--policy", policy_path, "-d"])
        out = capsys.readouterr().out
        # debug output includes "the policy creates by itself"
        assert "the policy creates by itself" in out

    def test_policy_with_no_created_types(self, write_policy, mock_seinfo, capsys):
        """Policy that doesn't define any types before require block."""
        policy = textwrap.dedent("""\
            module myapp 1.0;

            require {
                type httpd_t, tmp_t;
            }

            allow httpd_t tmp_t:file read;
        """)
        policy_path = write_policy(policy)
        with mock_seinfo():
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "summary:" in out

    def test_summary_line_format(self, write_policy, mock_seinfo, capsys):
        policy_path = write_policy(SIMPLE_POLICY)
        with mock_seinfo():
            sct.main(["--policy", policy_path])
        out = capsys.readouterr().out
        assert "----" in out
        assert "required types" in out
        assert "removed" in out
        assert "created" in out
        assert "renamed" in out

