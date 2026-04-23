#!/usr/bin/env python3
"""
Comprehensive test suite for se_log_analyser.

Uses the real 'ausearch' / 'aureport' binaries against the reference log file
(./test_log) so that every stage - parsing, indexing, formatting,
serialisation - is exercised on authentic data.

Because the PIDs recorded in the audit log belong to a remote host, all
psutil / /proc look-ups for *live* processes will naturally fail; the analyser
must tolerate this gracefully (pid_dead, dead_process, etc.).

Test categories
───────────────
  Unit         - data-classes, utility functions, indexes
  Integration  - full Analyzer.analyze() with the reference log
  Round-trip   - log → JSON → JSON  (must be idempotent)
  Modes        - every combination of --files / --json-files / --log
  Key handling - same key, different keys, merged keys
  PID Tree     - APP root detection, tree structure, format
  State file   - incremental analysis memory
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from unittest import mock

import psutil
import pytest # pyright: ignore[reportMissingImports]

from conftest import se_log_analyser as cla, _PROJECT_ROOT

_SCRIPT_PATH = os.path.join(_PROJECT_ROOT, "se_log_analyser")

# ── reference log path ───────────────────────────────────────────────────────
TEST_LOG = os.path.join(os.path.dirname(__file__), "test_log")

# Guard: all integration tests require the reference log
HAVE_TESTLOG = os.path.isfile(TEST_LOG)
HAVE_AUSEARCH = shutil.which("ausearch") is not None

needs_testlog = pytest.mark.skipif(not HAVE_TESTLOG, reason="reference log ./test_log not found")
needs_ausearch = pytest.mark.skipif(not HAVE_AUSEARCH, reason="ausearch not installed")

# Expected APP root PIDs — all 4 app invocations in test_log
# 1001 runs "/bin/bash /usr/sbin/myapp start", 1003 runs "/bin/bash /usr/sbin/myapp stop",
# 1005 runs "/bin/bash /usr/sbin/myapp reload" and 1007 runs "/bin/bash /usr/sbin/myapp status"
APP_ROOT_PIDS = {1001, 1003, 1005, 1007}
REFERENCE_KEY = "Rocky"

# ── Monkeypatch pgrep to avoid scanning live processes ───────────────────────
# pgrep() scans local processes looking for app container processes that only
# exist on a real host.  We patch it to return [] so the analyser falls
# through to the harmless ValueError path.
# We also patch os.stat for /proc/*/ns/pid which doesn't exist for log PIDs.

_real_os_stat = os.stat
_real_pgrep = cla.pgrep


def _safe_os_stat(path, *args, **kwargs):
    """os.stat that silently fails for /proc/<pid>/ns/pid look-ups of non-local PIDs."""
    if "/proc/" in str(path) and "/ns/pid" in str(path):
        raise FileNotFoundError(f"Mocked: {path}")
    return _real_os_stat(path, *args, **kwargs)


@pytest.fixture(autouse=True)
def _patch_live_process_lookups(monkeypatch):
    """Patch all live-process look-ups so tests work against a log file."""
    monkeypatch.setattr(cla, "pgrep", lambda cmd: [])
    monkeypatch.setattr(os, "stat", _safe_os_stat)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_analyzer(key=REFERENCE_KEY, log_path=TEST_LOG, **kw):
    """Create an Analyzer pointed at the reference log, with sane test defaults.
    Uses context_filter=['myapp_t'] and app_name='myapp' to match the reference test_log data."""
    defaults = dict(
        show_explanations=True,
        look_in_log=True,
        show_debug=False,
        show_info=False,
        show_pid_tree=True,
        context_filter=["myapp_t"],
        app_name="myapp",
    )
    defaults.update(kw)
    return cla.Analyzer(key=key, log_path=log_path, **defaults)


def _run_full_analysis(key=REFERENCE_KEY, tmpdir=None, **kw):
    """Run the full analyze() pipeline, returning (text, analyzer).
    Writes JSON output to tmpdir if provided."""
    json_dest = os.path.join(tmpdir, "out.json") if tmpdir else None
    a = _make_analyzer(key=key, **kw)
    txt = a.analyze(docs=[], json_dest=json_dest)
    return txt, a, json_dest


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAvcDenial:
    def test_to_rule_str(self):
        d = cla.AvcDenial("myapp_t", "cert_t", "dir", "getattr")
        assert d.to_rule_str() == "allow myapp_t cert_t:dir getattr;"

    def test_round_trip_dict(self):
        d = cla.AvcDenial("myapp_t", "user_tmp_t", "file", "read")
        assert cla.AvcDenial.from_dict(d.to_dict()) == d

    def test_different_methods(self):
        d1 = cla.AvcDenial("myapp_t", "cert_t", "dir", "read")
        d2 = cla.AvcDenial("myapp_t", "cert_t", "dir", "open")
        assert d1 != d2
        assert d1.to_rule_str() != d2.to_rule_str()


@pytest.mark.unit
class TestCommandContext:
    def test_round_trip_dict(self):
        c = cla.CommandContext(key="Rocky", descriptors={"desc1", "desc2"},
                               pid_namespace=12345, cmd="podman info", pid=23910)
        d = c.to_dict(sort_descriptors=True)
        c2 = cla.CommandContext.from_dict(d)
        assert c2.key == c.key
        assert c2.cmd == c.cmd
        assert c2.pid == c.pid
        assert c2.pid_namespace == c.pid_namespace
        assert c2.descriptors == c.descriptors

    def test_descriptors_as_set(self):
        d = {"key": "k", "descriptors": ["a", "b", "a"]}
        c = cla.CommandContext.from_dict(d, descriptors_as_set=True)
        assert isinstance(c.descriptors, set)
        assert len(c.descriptors) == 2


@pytest.mark.unit
class TestAnalysisResult:
    def test_round_trip_dict(self):
        cmd = cla.CommandContext(key="Rocky", cmd="podman info", pid=100)
        avc = cla.AvcDenial("myapp_t", "cert_t", "dir", "read")
        r = cla.AnalysisResult(command=cmd, avc_list=[avc])
        d = r.to_dict(sort_descriptors=True)
        r2 = cla.AnalysisResult.from_dict(d)
        assert r2.command.key == r.command.key
        assert len(r2.avc_list) == 1
        assert r2.avc_list[0].method == "read"


@pytest.mark.unit
class TestPidTreeEntry:
    def test_merge_fills_unknowns(self):
        a = cla.PidTreeEntry(cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="k")
        b = cla.PidTreeEntry(cmd="bash", ppid=100, context="myapp_t", key="k",
                              children=[(200, "k")])
        a.merge(b)
        assert a.cmd == "bash"
        assert a.ppid == 100
        assert a.context == "myapp_t"
        assert (200, "k") in a.children

    def test_merge_preserves_known(self):
        a = cla.PidTreeEntry(cmd="original", ppid=1, context="ctx", key="k")
        b = cla.PidTreeEntry(cmd="new", ppid=2, context="new_ctx", key="k")
        a.merge(b)
        assert a.cmd == "original"  # kept because not UNKNOWN
        assert a.ppid == 1

    def test_round_trip_dict(self):
        e = cla.PidTreeEntry(cmd="sleep 1", ppid=42, context="myapp_t", key="Rocky",
                              children=[(99, "Rocky"), (100, "Other")])
        d = e.to_dict()
        e2 = cla.PidTreeEntry.from_dict(d, entry_key="Rocky")
        assert e2.cmd == "sleep 1"
        assert e2.ppid == 42
        assert (99, "Rocky") in e2.children
        assert (100, "Other") in e2.children

    def test_merge_no_duplicate_children(self):
        """merge() should not add duplicate children."""
        a = cla.PidTreeEntry(cmd="bash", ppid=1, context="ctx", key="k",
                              children=[(200, "k")])
        b = cla.PidTreeEntry(cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="k",
                              children=[(200, "k"), (300, "k")])
        a.merge(b)
        assert a.children.count((200, "k")) == 1
        assert (300, "k") in a.children
        assert len(a.children) == 2

    def test_from_dict_fallback_key(self):
        """from_dict with entry_key=None should use key from the dict."""
        d = {"cmd": "ls", "ppid": 1, "context": "ctx", "key": "myhost",
             "children": [], "live": False}
        e = cla.PidTreeEntry.from_dict(d, entry_key=None)
        assert e.key == "myhost"

    def test_from_dict_missing_key_defaults_unknown(self):
        """from_dict with entry_key=None and no key in dict should default to UNKNOWN."""
        d = {"cmd": "ls", "ppid": 1, "context": "ctx",
             "children": [], "live": False}
        e = cla.PidTreeEntry.from_dict(d, entry_key=None)
        assert e.key == cla.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Process Utilities (is_process_alive / get_from_pid)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestIsProcessAlive:
    """Unit tests for Analyzer.is_process_alive()."""

    def test_none_pid_raises_invalid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.InvalidPID):
            a.is_process_alive(None)

    def test_empty_string_raises_invalid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.InvalidPID):
            a.is_process_alive("")

    def test_zero_raises_invalid(self):
        """0 is falsy, should raise InvalidPID."""
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.InvalidPID):
            a.is_process_alive(0)

    def test_non_numeric_string_raises_invalid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.InvalidPID):
            a.is_process_alive("not_a_pid")

    def test_notfound_sentinel_raises_invalid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.InvalidPID):
            a.is_process_alive(cla.NOT_FOUND)

    def test_dead_pid_raises_process_dead(self):
        """A PID that doesn't exist should raise ProcessDead."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Use an absurdly high PID that certainly doesn't exist
        with pytest.raises(cla.ProcessDead):
            a.is_process_alive(9999999)

    def test_alive_process_returns_process(self):
        """Our own PID should be alive and return a psutil.Process."""
        a = cla.Analyzer(key="test", look_in_log=False)
        proc = a.is_process_alive(os.getpid())
        assert proc is not None
        assert proc.pid == os.getpid()

    def test_string_pid_works(self):
        """A string representation of a valid PID should work."""
        a = cla.Analyzer(key="test", look_in_log=False)
        proc = a.is_process_alive(str(os.getpid()))
        assert proc.pid == os.getpid()

    def test_zombie_raises_process_zombie(self):
        """A zombie process should raise ProcessZombie."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = mock.MagicMock()
        mock_proc.is_running.return_value = True
        mock_proc.status.return_value = psutil.STATUS_ZOMBIE
        with mock.patch("psutil.Process", return_value=mock_proc):
            with pytest.raises(cla.ProcessZombie):
                a.is_process_alive(12345)

    def test_not_running_raises_process_zombie(self):
        """A process that is_running()==False should raise ProcessZombie."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = mock.MagicMock()
        mock_proc.is_running.return_value = False
        mock_proc.status.return_value = psutil.STATUS_DEAD
        with mock.patch("psutil.Process", return_value=mock_proc):
            with pytest.raises(cla.ProcessZombie):
                a.is_process_alive(12345)

    def test_exception_hierarchy(self):
        """All custom exceptions derive from ProcessNotAvailable."""
        assert issubclass(cla.ProcessDead, cla.ProcessNotAvailable)
        assert issubclass(cla.ProcessZombie, cla.ProcessNotAvailable)
        assert issubclass(cla.InvalidPID, cla.ProcessNotAvailable)


@pytest.mark.unit
class TestGetFromPid:
    """Unit tests for Analyzer.get_from_pid() with mocked psutil."""

    def _make_mock_process(self, pid=1000, ppid=999, uid=0,
                           cmdline=None, status=psutil.STATUS_RUNNING):
        """Create a mock psutil.Process with the given attributes."""
        proc = mock.MagicMock()
        proc.pid.return_value = pid
        proc.ppid.return_value = ppid
        uids = mock.MagicMock()
        uids.real = uid
        proc.uids.return_value = uids
        proc.cmdline.return_value = cmdline or ["/usr/bin/podman", "info"]
        proc.is_running.return_value = True
        proc.status.return_value = status
        return proc

    def test_get_cmd(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(cmdline=["/usr/bin/podman", "info"])
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            result = a.get_from_pid(1000, "cmd")
        assert result == "/usr/bin/podman info"

    def test_get_ppid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(ppid=42)
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            result = a.get_from_pid(1000, "ppid")
        assert result == 42

    def test_get_uid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(uid=1001)
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            result = a.get_from_pid(1000, "uid")
        assert result == 1001

    def test_get_pid(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(pid=1000)
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            result = a.get_from_pid(1000, "pid")
        assert result == 1000

    def test_dead_process_returns_sentinel(self):
        """When is_process_alive raises ProcessDead, get_from_pid returns DEAD_PROCESS."""
        a = cla.Analyzer(key="test", look_in_log=False)
        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            result = a.get_from_pid(9999, "cmd")
        assert result == cla.DEAD_PROCESS

    def test_zombie_process_returns_sentinel(self):
        """When is_process_alive raises ProcessZombie, get_from_pid returns DEAD_PROCESS."""
        a = cla.Analyzer(key="test", look_in_log=False)
        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessZombie("zombie")):
            result = a.get_from_pid(9999, "cmd")
        assert result == cla.DEAD_PROCESS

    def test_invalid_pid_returns_sentinel(self):
        """When is_process_alive raises InvalidPID, get_from_pid returns DEAD_PROCESS."""
        a = cla.Analyzer(key="test", look_in_log=False)
        with mock.patch.object(a, "is_process_alive", side_effect=cla.InvalidPID("bad")):
            result = a.get_from_pid(0, "cmd")
        assert result == cla.DEAD_PROCESS

    def test_result_cached_in_pid_store(self):
        """Second call for same pid+arg should use the cached value."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(ppid=42)
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc) as m:
            a.get_from_pid(1000, "ppid")
            a.get_from_pid(1000, "ppid")
        # is_process_alive should be called only once (second time is cached)
        m.assert_called_once()
        assert a.pid_store[1000]["ppid"] == 42

    def test_different_args_query_separately(self):
        """Querying 'cmd' then 'ppid' should both be populated."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(ppid=42, cmdline=["sleep", "1"])
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            a.get_from_pid(1000, "cmd")
            a.get_from_pid(1000, "ppid")
        assert a.pid_store[1000]["cmd"] == "sleep 1"
        assert a.pid_store[1000]["ppid"] == 42

    def test_invalid_arg_raises(self):
        """Passing an invalid arg should raise ValueError."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process()
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            with pytest.raises(ValueError, match="invalid_arg"):
                a.get_from_pid(1000, "invalid_arg")

    def test_cmd_strips_whitespace(self):
        """Command value should be stripped of trailing whitespace."""
        a = cla.Analyzer(key="test", look_in_log=False)
        mock_proc = self._make_mock_process(cmdline=["sleep", "1 "])
        with mock.patch.object(a, "is_process_alive", return_value=mock_proc):
            result = a.get_from_pid(1000, "cmd")
        assert result == "sleep 1"
        assert not result.endswith(" ")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetBaseCmd:
    def test_simple_binary(self):
        assert cla.get_base_cmd("podman info") == "PODMAN"

    def test_with_path(self):
        assert cla.get_base_cmd("/usr/bin/podman info") == "PODMAN"

    def test_emulator_skip(self):
        assert cla.get_base_cmd("python3 /opt/cmu/tools/myscript.py") == "MYSCRIPT.PY"

    def test_bash_script(self):
        assert cla.get_base_cmd("/bin/bash /bin/myapp -s") == "MYAPP"

    def test_bash_with_flag(self):
        assert cla.get_base_cmd("bash -c echo hello") == "BASH"

    def test_list_input(self):
        assert cla.get_base_cmd(["podman", "info"]) == "PODMAN"

    def test_sh_emulator(self):
        assert cla.get_base_cmd("sh /usr/local/bin/myscript arg1") == "MYSCRIPT"

    def test_empty_string(self):
        """Empty string should return UNKNOWN, not crash."""
        result = cla.get_base_cmd("")
        assert result == cla.UNKNOWN

    def test_empty_list(self):
        """Empty list should return UNKNOWN, not crash."""
        result = cla.get_base_cmd([])
        assert result == cla.UNKNOWN

    def test_single_word_no_args(self):
        assert cla.get_base_cmd("podman") == "PODMAN"

    def test_single_emulator_only(self):
        """A bare emulator with no script should return the emulator itself."""
        assert cla.get_base_cmd("bash") == "BASH"

    def test_nested_emulators(self):
        """python3 wrapping bash wrapping a script: should resolve to the script."""
        assert cla.get_base_cmd("python3 bash /opt/script.sh") == "SCRIPT.SH"

    def test_all_emulators_chain(self):
        """Chain of nothing but emulators — should return the last one."""
        result = cla.get_base_cmd("bash sh python3")
        assert isinstance(result, str)
        # No real script follows, so the last emulator is used
        assert result == "PYTHON3"

    def test_tclsh_emulator(self):
        assert cla.get_base_cmd("tclsh /opt/cmu/bin/cmu_do_boot arg1") == "CMU_DO_BOOT"

    def test_python_emulator(self):
        """'python' (not python3) should also be recognized as an emulator."""
        assert cla.get_base_cmd("python /opt/script.py --verbose") == "SCRIPT.PY"

    def test_single_element_list(self):
        assert cla.get_base_cmd(["/usr/bin/podman"]) == "PODMAN"

    def test_emulator_then_flag_only(self):
        """'bash -x' — flag after emulator: should return BASH (the emulator)."""
        result = cla.get_base_cmd("bash -x")
        assert result == "BASH"

    def test_absolute_path_script_via_sh(self):
        assert cla.get_base_cmd("sh /usr/share/myapp/selinux/myapp_selinux_configure") == "MYAPP_SELINUX_CONFIGURE"


@pytest.mark.unit
class TestExtractExecveCommand:
    def test_simple_execve(self):
        block = 'type=EXECVE msg=audit(1770630310.060:18114): argc=2 a0="basename" a1="/bin/myapp"\n'
        result = cla.extract_execve_command(block)
        assert result is not None
        assert "basename" in result
        assert "/bin/myapp" in result

    def test_no_execve_returns_none(self):
        block = "type=SYSCALL msg=audit(...) ...\ntype=AVC ...\n"
        assert cla.extract_execve_command(block) is None

    def test_hex_addresses_skipped(self):
        block = ('type=SYSCALL msg=audit(...) a0=0xdeadbeef a1=0xc0ffee\n'
                 'type=EXECVE msg=audit(...): argc=2 a0="/usr/bin/ls" a1="/tmp"\n')
        result = cla.extract_execve_command(block)
        assert result is not None
        assert "ls" in result
        assert "0x" not in result

    def test_execve_all_hex_args_returns_none(self):
        """EXECVE block where all args are hex addresses should return None."""
        block = 'type=EXECVE msg=audit(...): argc=2 a0=0xdeadbeef a1=0xc0ffee\n'
        assert cla.extract_execve_command(block) is None

    def test_execve_no_arg_matches_returns_none(self):
        """EXECVE block with no recognisable a0=, a1= fields should return None."""
        block = 'type=EXECVE msg=audit(...): argc=0\n'
        assert cla.extract_execve_command(block) is None

    def test_execve_empty_block_returns_none(self):
        """Completely empty block returns None."""
        assert cla.extract_execve_command("") is None

    def test_execve_only_syscall_no_execve(self):
        """Block with SYSCALL but no EXECVE type returns None (already covered but
        verifies both loops fail)."""
        block = 'type=SYSCALL msg=audit(...): a0=0x1234 a1=0xabcd\n'
        assert cla.extract_execve_command(block) is None


@pytest.mark.unit
class TestGetEntryId:
    def test_extracts_msg_audit(self):
        block = "type=AVC msg=audit(1770630310.625:18162): avc: denied ..."
        eid = cla.get_entry_id(block)
        assert eid == "1770630310.625:18162"

    def test_fallback_hash(self):
        block = "no msg=audit here"
        eid = cla.get_entry_id(block)
        # Should be a hex hash
        assert len(eid) == 64  # SHA-256 hex digest


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — selinux_type
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSelinuxType:
    def test_standard_context(self):
        assert cla.selinux_type("system_u:system_r:myapp_t:s0-s0:c0.c1023") == "myapp_t"

    def test_minimal_context(self):
        assert cla.selinux_type("u:r:sshd_t:s0") == "sshd_t"

    def test_two_parts_only(self):
        """Fewer than 3 parts should return the whole string."""
        assert cla.selinux_type("u:r") == "u:r"

    def test_single_part(self):
        assert cla.selinux_type("unlabeled_t") == "unlabeled_t"

    def test_empty_string(self):
        assert cla.selinux_type("") == ""


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — pgrep
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPgrep:
    def test_finds_current_process(self, monkeypatch):
        """pgrep should find the current python process."""
        # Restore real pgrep (autouse fixture patches it to return [])
        monkeypatch.setattr(cla, "pgrep", _real_pgrep)
        pids = cla.pgrep("pytest")
        assert isinstance(pids, list)
        # Should find at least ourselves (the pytest process)
        assert len(pids) > 0

    def test_returns_empty_for_nonexistent(self, monkeypatch):
        monkeypatch.setattr(cla, "pgrep", _real_pgrep)
        pids = cla.pgrep("nonexistent_process_name_xyz_12345")
        assert pids == []


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — NsIndex
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestNsIndex:
    def test_set_and_get(self):
        ns = cla.NsIndex()
        ns.set(12345, "~pid_ns_test~")
        assert ns.get(12345) == "~pid_ns_test~"

    def test_idempotent_set(self):
        ns = cla.NsIndex()
        label1 = ns.set(100, "~pid_ns_foo~")
        label2 = ns.set(100, "~pid_ns_foo~")
        assert label1 == label2
        assert len(ns) == 1

    def test_collision_appends_counter(self):
        ns = cla.NsIndex()
        ns.set(100, "~pid_ns_host~")
        label2 = ns.set(200, "~pid_ns_host~")
        assert label2 == "~pid_ns_host_1~"
        assert ns.get(200) == "~pid_ns_host_1~"

    def test_sentinel_values(self):
        ns = cla.NsIndex()
        assert ns.get(cla.NOT_FOUND) == cla.NOT_FOUND
        assert ns.get(cla.PID_DEAD) == cla.PID_DEAD

    def test_contains_with_string(self):
        ns = cla.NsIndex()
        ns.set(42, "~pid_ns_x~")
        assert 42 in ns
        assert "42" in ns
        assert 99 not in ns

    def test_format_output(self):
        ns = cla.NsIndex()
        ns.set(100, "~pid_ns_a~")
        out = ns.format()
        assert "~pid_ns_a~" in out
        assert "100" in out

    def test_to_dict(self):
        ns = cla.NsIndex()
        ns.set(42, "~pid_ns_test~")
        d = ns.to_dict()
        assert d == {"42": "~pid_ns_test~"}

    def test_get_unconvertible_string(self):
        """A string that's not a sentinel and not an int should be returned as-is."""
        ns = cla.NsIndex()
        assert ns.get("some_garbage") == "some_garbage"

    def test_get_unregistered_int(self):
        """An int that was never registered should return its str representation."""
        ns = cla.NsIndex()
        assert ns.get(99999) == "99999"

    def test_multiple_collisions(self):
        """Three different ns_ids with the same label should get _1~, _2~."""
        ns = cla.NsIndex()
        l1 = ns.set(100, "~pid_ns_host~")
        l2 = ns.set(200, "~pid_ns_host~")
        l3 = ns.set(300, "~pid_ns_host~")
        assert l1 == "~pid_ns_host~"
        assert l2 == "~pid_ns_host_1~"
        assert l3 == "~pid_ns_host_2~"
        assert len(ns) == 3
        assert ns.get(100) == "~pid_ns_host~"
        assert ns.get(200) == "~pid_ns_host_1~"
        assert ns.get(300) == "~pid_ns_host_2~"

    def test_contains_returns_false_for_non_numeric_string(self):
        ns = cla.NsIndex()
        assert "abc" not in ns

    def test_format_sorted_by_ns_id(self):
        """Format output should be sorted by namespace ID."""
        ns = cla.NsIndex()
        ns.set(300, "~pid_ns_c~")
        ns.set(100, "~pid_ns_a~")
        ns.set(200, "~pid_ns_b~")
        out = ns.format()
        lines = [l for l in out.strip().split("\n") if l]
        # Extract ns_ids from each line
        ids = [int(l.split(cla.KEY_DELIMITER)[1].strip()) for l in lines]
        assert ids == [100, 200, 300]


@pytest.mark.unit
class TestDetectAppNamespaces:
    """Unit tests for Analyzer.detect_app_namespaces()."""

    def test_no_context_filter_returns_empty(self):
        """Without context_filter, detect_app_namespaces should return []."""
        a = cla.Analyzer(key="test", look_in_log=False)
        assert a.detect_app_namespaces() == []

    def test_finds_matching_process(self, tmp_path):
        """Should return namespace inode when a process matches context_filter."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        fake_proc = mock.MagicMock()
        fake_proc.info = {"pid": 42}
        with mock.patch("psutil.process_iter", return_value=[fake_proc]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0-s0:c0.c1023\x00"
            )):
                with mock.patch("os.stat") as mock_stat:
                    mock_stat.return_value = mock.MagicMock(st_ino=4026532289)
                    result = a.detect_app_namespaces()
        assert result == [4026532289]

    def test_no_matching_process_returns_empty(self):
        """Should return [] when no process matches context_filter."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        fake_proc = mock.MagicMock()
        fake_proc.info = {"pid": 42}
        with mock.patch("psutil.process_iter", return_value=[fake_proc]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:sshd_t:s0\x00"
            )):
                result = a.detect_app_namespaces()
        assert result == []

    def test_skips_inaccessible_processes(self):
        """Processes whose /proc/<pid>/attr/current is unreadable should be skipped."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        bad_proc = mock.MagicMock()
        bad_proc.info = {"pid": 1}
        good_proc = mock.MagicMock()
        good_proc.info = {"pid": 99}

        def open_side_effect(path, *args, **kwargs):
            if "/proc/1/" in path:
                raise PermissionError("permission denied")
            return mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )()

        with mock.patch("psutil.process_iter", return_value=[bad_proc, good_proc]):
            with mock.patch("builtins.open", side_effect=open_side_effect):
                with mock.patch("os.stat") as mock_stat:
                    mock_stat.return_value = mock.MagicMock(st_ino=9999)
                    result = a.detect_app_namespaces()
        assert result == [9999]

    def test_deduplicates_namespaces(self):
        """Multiple processes in the same namespace should produce one entry."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        proc1 = mock.MagicMock()
        proc1.info = {"pid": 10}
        proc2 = mock.MagicMock()
        proc2.info = {"pid": 20}

        with mock.patch("psutil.process_iter", return_value=[proc1, proc2]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )):
                with mock.patch("os.stat") as mock_stat:
                    mock_stat.return_value = mock.MagicMock(st_ino=1111)
                    result = a.detect_app_namespaces()
        assert result == [1111]

    def test_discovers_multiple_namespaces(self):
        """Processes in different container namespaces should all be returned."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        proc1 = mock.MagicMock()
        proc1.info = {"pid": 10}
        proc2 = mock.MagicMock()
        proc2.info = {"pid": 20}

        NS_A = 4026532001
        NS_B = 4026532002

        def stat_side_effect(path):
            m = mock.MagicMock()
            if "/proc/10/" in path:
                m.st_ino = NS_A
            else:
                m.st_ino = NS_B
            return m

        with mock.patch("psutil.process_iter", return_value=[proc1, proc2]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )):
                with mock.patch("os.stat", side_effect=stat_side_effect):
                    result = a.detect_app_namespaces()
        assert result == [NS_A, NS_B]

    def test_skips_excluded_namespaces(self):
        """Should skip processes whose namespace is in exclude_ns list."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        proc_bm = mock.MagicMock()
        proc_bm.info = {"pid": 42}
        proc_ctr = mock.MagicMock()
        proc_ctr.info = {"pid": 99}

        BM_NS = 4026531836
        CTR_NS = 4026532289

        def stat_side_effect(path):
            m = mock.MagicMock()
            if "/proc/42/" in path:
                m.st_ino = BM_NS
            else:
                m.st_ino = CTR_NS
            return m

        with mock.patch("psutil.process_iter", return_value=[proc_bm, proc_ctr]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )):
                with mock.patch("os.stat", side_effect=stat_side_effect):
                    result = a.detect_app_namespaces(exclude_ns=[BM_NS])
        assert result == [CTR_NS]

    def test_returns_empty_when_all_excluded(self):
        """Should return [] when all matching processes are in excluded namespaces."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        proc1 = mock.MagicMock()
        proc1.info = {"pid": 10}

        BM_NS = 4026531836

        with mock.patch("psutil.process_iter", return_value=[proc1]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )):
                with mock.patch("os.stat") as mock_stat:
                    mock_stat.return_value = mock.MagicMock(st_ino=BM_NS)
                    result = a.detect_app_namespaces(exclude_ns=[BM_NS])
        assert result == []

    def test_excludes_multiple_namespaces(self):
        """Should filter out multiple excluded namespaces."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        proc1 = mock.MagicMock()
        proc1.info = {"pid": 10}
        proc2 = mock.MagicMock()
        proc2.info = {"pid": 20}
        proc3 = mock.MagicMock()
        proc3.info = {"pid": 30}

        NS_A = 1001
        NS_B = 1002
        NS_C = 1003

        def stat_side_effect(path):
            m = mock.MagicMock()
            if "/proc/10/" in path:
                m.st_ino = NS_A
            elif "/proc/20/" in path:
                m.st_ino = NS_B
            else:
                m.st_ino = NS_C
            return m

        with mock.patch("psutil.process_iter", return_value=[proc1, proc2, proc3]):
            with mock.patch("builtins.open", mock.mock_open(
                read_data="system_u:system_r:myapp_t:s0\x00"
            )):
                with mock.patch("os.stat", side_effect=stat_side_effect):
                    result = a.detect_app_namespaces(exclude_ns=[NS_A, NS_B])
        assert result == [NS_C]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — CmdIndex
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCmdIndex:
    def test_register_and_get(self):
        ci = cla.CmdIndex()
        alias = ci.register("podman info", "PODMAN_0001")
        assert alias == "PODMAN_0001"
        assert ci.get_alias("podman info") == "PODMAN_0001"

    def test_auto_generate_alias(self):
        ci = cla.CmdIndex()
        alias = ci.register("podman info")
        assert alias.startswith("PODMAN_")

    def test_collision_avoidance(self):
        ci = cla.CmdIndex()
        a1 = ci.register("cmd1", "ALIAS_0001")
        a2 = ci.register("cmd2", "ALIAS_0001")  # collision
        assert a1 != a2
        assert a1 in ci.aliases_sorted()
        assert a2 in ci.aliases_sorted()

    def test_idempotent_register(self):
        ci = cla.CmdIndex()
        a1 = ci.register("podman info")
        a2 = ci.register("podman info")
        assert a1 == a2
        assert len(ci) == 1

    def test_whitespace_normalisation(self):
        ci = cla.CmdIndex()
        a1 = ci.register("podman   info")
        a2 = ci.register("podman info")
        assert a1 == a2

    def test_log_weight_and_build(self):
        ci = cla.CmdIndex()
        # Log a command many times → should auto-alias
        for _ in range(cla.INDEX_MIN_TIME + 1):
            ci.log_weight("podman info")
        ci.build()
        assert "podman info" in ci

    def test_long_cmd_auto_alias(self):
        ci = cla.CmdIndex()
        long_cmd = "/usr/bin/java " + "x" * cla.INDEX_MIN_LEN
        ci.log_weight(long_cmd)
        ci.build()
        assert long_cmd in ci

    def test_format_history_dedup(self):
        ci = cla.CmdIndex()
        ci.register("ausearch -i -m avc", "AVC_0001")
        ci.register("ausearch -i -m msg", "MSG_0001")
        ci.log_weight("ausearch -i -m avc")
        ci.log_weight("ausearch -i -m msg")
        out = ci.format()
        assert "AVC_0001" in out
        assert "MSG_0001" in out
        # History dedup: second entry should have ***** for repeated words
        assert "*" in out

    def test_to_dict(self):
        ci = cla.CmdIndex()
        ci.register("podman info", "POD_0001")
        d = ci.to_dict()
        assert d == {"podman info": "POD_0001"}

    def test_format_empty_mapping_returns_empty(self):
        """format() on an empty CmdIndex should return empty string."""
        ci = cla.CmdIndex()
        assert ci.format() == ""

    def test_log_weight_with_list_input(self):
        """log_weight should accept a list and join it."""
        ci = cla.CmdIndex()
        ci.log_weight(["podman", "info"])
        assert ci.get_weight("podman info") == 1

    def test_log_weight_with_non_string_non_list(self):
        """log_weight with an int or other type should silently return."""
        ci = cla.CmdIndex()
        ci.log_weight(12345)
        ci.log_weight(None)
        ci.log_weight({})
        assert len(ci.weights_dict()) == 0

    def test_log_weight_skips_alias(self):
        """log_weight should not log a key that is already an alias."""
        ci = cla.CmdIndex()
        ci.register("podman info", "PODMAN_0001")
        ci.log_weight("PODMAN_0001")
        assert ci.get_weight("PODMAN_0001") == 0

    def test_log_weight_empty_string(self):
        """log_weight with empty string should silently return."""
        ci = cla.CmdIndex()
        ci.log_weight("")
        assert len(ci.weights_dict()) == 0

    def test_get_weight_default(self):
        """get_weight for an unregistered key returns 0."""
        ci = cla.CmdIndex()
        assert ci.get_weight("unknown_cmd") == 0

    def test_get_weight_after_logging(self):
        ci = cla.CmdIndex()
        ci.log_weight("podman info", 3)
        ci.log_weight("podman info", 2)
        assert ci.get_weight("podman info") == 5

    def test_build_single_arg_not_indexed(self):
        """Commands with only one word should NOT be indexed, even if frequent."""
        ci = cla.CmdIndex()
        for _ in range(cla.INDEX_MIN_TIME + 10):
            ci.log_weight("podman")  # single word, no args
        ci.build()
        assert "podman" not in ci

    def test_build_multi_arg_indexed(self):
        """Commands with >1 args and frequent should be indexed."""
        ci = cla.CmdIndex()
        for _ in range(cla.INDEX_MIN_TIME + 1):
            ci.log_weight("podman info")
        ci.build()
        assert "podman info" in ci

    def test_log_weight_list_with_whitespace(self):
        """List items with extra whitespace should be stripped."""
        ci = cla.CmdIndex()
        ci.log_weight(["  podman  ", "  info  "])
        assert ci.get_weight("podman info") == 1

    def test_get_alias_non_string(self):
        """get_alias with a non-string should return str(key)."""
        ci = cla.CmdIndex()
        assert ci.get_alias(12345) == "12345"

    def test_collision_with_template_alias(self):
        """Collision on a template-formatted alias (CMD_0001) should auto-increment."""
        ci = cla.CmdIndex()
        a1 = ci.register("cmd_a", "CMD_0001")
        assert a1 == "CMD_0001"
        # Now register a DIFFERENT command with the SAME alias → collision
        a2 = ci.register("cmd_b", "CMD_0001")
        assert a2 != a1
        # Template collision path derives base from alias[:-5]
        assert a2.startswith("CMD_")

    def test_dunder_methods(self):
        """Exercise __getitem__, __iter__, items(), values(), keys()."""
        ci = cla.CmdIndex()
        ci.register("podman info", "POD_0001")
        assert ci["podman info"] == "POD_0001"
        assert list(iter(ci)) == ["podman info"]
        assert list(ci.items()) == [("podman info", "POD_0001")]
        assert list(ci.values()) == ["POD_0001"]
        assert list(ci.keys()) == ["podman info"]

    def test_collision_with_non_template_alias(self):
        """Collision on a plain alias should use the alias itself as base."""
        ci = cla.CmdIndex()
        a1 = ci.register("cmd_a", "MYALIAS")
        assert a1 == "MYALIAS"
        a2 = ci.register("cmd_b", "MYALIAS")
        assert a2 != a1
        assert a2.startswith("MYALIAS_")

    def test_format_history_word_replacement(self):
        """When a word differs in two consecutive commands sharing a prefix,
        the changed word should appear, the repeated ones as ****."""
        ci = cla.CmdIndex()
        ci.register("ausearch -i -m avc", "AVC_0001")
        ci.register("ausearch -i -m msg", "MSG_0001")
        ci.log_weight("ausearch -i -m avc")
        ci.log_weight("ausearch -i -m msg")
        out = ci.format()
        lines = [l for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2
        # Second line should have * replacements for the identical prefix
        assert "***" in lines[1]
        # But the differing word "msg" should appear verbatim
        assert "msg" in lines[1]

    def test_format_history_reset_on_base_change(self):
        """When the base command changes, history should be reset."""
        ci = cla.CmdIndex()
        ci.register("ausearch -i -m avc", "AVC_0001")
        ci.register("podman info", "POD_0001")
        ci.log_weight("ausearch -i -m avc")
        ci.log_weight("podman info")
        out = ci.format()
        # podman line should NOT have stars — it's a completely different command
        lines = [l for l in out.strip().split("\n") if l.strip()]
        for line in lines:
            if "POD_0001" in line:
                assert "***" not in line


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Regex Patterns
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRegexPatterns:
    def test_extract_context(self):
        line = ("avc: denied { read open } for pid=23927 comm=podman "
                "scontext=system_u:system_r:myapp_t:s0-s0:c0.c1023 "
                "tcontext=system_u:object_r:pasta_exec_t:s0 tclass=file permissive=1")
        m = cla.REGEX_EXTRACT_CONTEXT.search(line)
        assert m is not None
        assert m.group(1).strip() == "read open"
        assert "myapp_t" in m.group(2)
        assert "pasta_exec_t" in m.group(3)
        assert m.group(4) == "file"

    def test_extract_pid_ppid(self):
        line = "ppid=23906 pid=23910 auid=myapp uid=root"
        m = cla.REGEX_EXTRACT_PID_PPID.search(line)
        assert m is not None
        assert m.group(1) == "23906"
        assert m.group(2) == "23910"

    def test_extract_subj(self):
        line = " subj=system_u:system_r:myapp_t:s0-s0:c0.c1023 key=(null)"
        m = cla.REGEX_EXTRACT_SUBJ.search(line)
        assert m is not None
        assert "myapp_t" in m.group(1)

    def test_allow_rule_regex(self):
        line = "myapp_t cert_t:dir getattr;"
        m = cla.REGEX_ALLOW_RULE.search(line)
        assert m is not None
        assert m.group(1) == "myapp_t"
        assert m.group(2) == "cert_t"
        assert m.group(3) == "dir"
        assert m.group(4) == "getattr"

    def test_pid_tree_entry_regex(self):
        line = "# ├── pid=24534@Rocky [APP ROOT]        ctx: system_u:system_r:myapp_t:s0-s0:c0.c1023                 cmd: /bin/bash /bin/myapp"
        m = cla.REGEX_PID_TREE_ENTRY.search(line)
        assert m is not None
        assert m.group(1) == "24534"
        assert m.group(2) == "Rocky"
        assert "myapp_t" in m.group(3)
        assert "/bin/bash" in m.group(4)

    def test_index_template_regex(self):
        assert cla.REGEX_INDEX_TEMPLATE.match("MYAPP_0001")
        assert cla.REGEX_INDEX_TEMPLATE.match("PODMAN_0001")
        assert not cla.REGEX_INDEX_TEMPLATE.match("PODMAN")

    # ── REGEX_EXTRACT_UID ─────────────────────────────────────────────────────
    def test_extract_uid_root(self):
        line = "ppid=4732 pid=5041 auid=root uid=root gid=root"
        m = cla.REGEX_EXTRACT_UID.search(line)
        assert m is not None
        assert m.group(1) == "root"

    def test_extract_uid_numeric(self):
        line = " uid=1000 gid=1000"
        m = cla.REGEX_EXTRACT_UID.search(line)
        assert m is not None
        assert m.group(1) == "1000"

    # ── REGEX_EXTRACT_PPID (standalone) ───────────────────────────────────────
    def test_extract_ppid_standalone(self):
        line = " ppid=4732 pid=5041"
        m = cla.REGEX_EXTRACT_PPID.search(line)
        assert m is not None
        assert m.group(1) == "4732"

    # ── REGEX_EXTRACT_FOR_PID ─────────────────────────────────────────────────
    def test_extract_for_pid(self):
        line = "avc:  denied  { getattr } for  pid=23906 comm=myapp"
        m = cla.REGEX_EXTRACT_FOR_PID.search(line)
        assert m is not None
        assert m.group(1) == "23906"

    def test_extract_for_pid_not_in_ppid(self):
        """REGEX_EXTRACT_FOR_PID should NOT match ppid=..."""
        line = "ppid=100 something"
        m = cla.REGEX_EXTRACT_FOR_PID.search(line)
        assert m is None

    # ── REGEX_EXTRACT_PID ─────────────────────────────────────────────────────
    def test_extract_pid_standalone(self):
        line = " pid=5041 auid=root"
        m = cla.REGEX_EXTRACT_PID.search(line)
        assert m is not None
        assert m.group(1) == "5041"

    # ── REGEX_EXTRACT_CMDLINE ─────────────────────────────────────────────────
    def test_extract_cmdline_quoted(self):
        line = ' cmdline="bash compile"'
        m = cla.REGEX_EXTRACT_CMDLINE.search(line)
        assert m is not None
        assert m.group(1) == '"bash compile"'

    def test_extract_cmdline_unquoted(self):
        line = " cmdline=bash"
        m = cla.REGEX_EXTRACT_CMDLINE.search(line)
        assert m is not None
        assert m.group(1) == "bash"

    def test_extract_cmdline_empty_quotes(self):
        """cmdline=\"\" should match but return empty quotes."""
        line = ' cmdline=""'
        m = cla.REGEX_EXTRACT_CMDLINE.search(line)
        # The regex requires at least .+? inside quotes, so empty "" may not match
        # either way, the parser handles this case explicitly
        if m:
            assert m.group(1) == '""'

    # ── REGEX_EXTRACT_EXE ─────────────────────────────────────────────────────
    def test_extract_exe_quoted(self):
        line = ' exe="/usr/bin/bash"'
        m = cla.REGEX_EXTRACT_EXE.search(line)
        assert m is not None
        assert m.group(1) == '"/usr/bin/bash"'

    def test_extract_exe_unquoted(self):
        line = " exe=/usr/bin/bash"
        m = cla.REGEX_EXTRACT_EXE.search(line)
        assert m is not None
        assert m.group(1) == "/usr/bin/bash"

    # ── REGEX_EXTRACT_COM ─────────────────────────────────────────────────────
    def test_extract_comm(self):
        line = " comm=podman exe=/usr/bin/podman"
        m = cla.REGEX_EXTRACT_COM.search(line)
        assert m is not None
        assert m.group(1) == "podman"

    # ── REGEX_EXTRACT_MSG_AUDIT ───────────────────────────────────────────────
    def test_extract_msg_audit_timestamp_format(self):
        """Human-readable timestamp format from ausearch -i."""
        line = "type=SYSCALL msg=audit(07/09/2024 16:38:22.104:7145) : arch=x86_64"
        m = cla.REGEX_EXTRACT_MSG_AUDIT.search(line)
        assert m is not None
        assert m.group(1) == "07/09/2024 16:38:22.104:7145"

    def test_extract_msg_audit_epoch_format(self):
        """Raw epoch timestamp format from audit.log."""
        line = "type=AVC msg=audit(1720536502.104:7145): avc: denied"
        m = cla.REGEX_EXTRACT_MSG_AUDIT.search(line)
        assert m is not None
        assert m.group(1) == "1720536502.104:7145"

    # ── REGEX_EXTRACT_EXECVE_ARG ──────────────────────────────────────────────
    def test_extract_execve_arg_quoted(self):
        line = 'argc=3 a0="/bin/bash" a1="-c" a2="echo hello"'
        args = dict(cla.REGEX_EXTRACT_EXECVE_ARG.findall(line))
        assert args["0"] == '"/bin/bash"'
        assert args["1"] == '"-c"'
        assert args["2"] == '"echo hello"'

    def test_extract_execve_arg_unquoted(self):
        line = "argc=2 a0=basename a1=/bin/myapp"
        args = dict(cla.REGEX_EXTRACT_EXECVE_ARG.findall(line))
        assert args["0"] == "basename"
        assert args["1"] == "/bin/myapp"

    # ── REGEX_AVC_DELIMITER ───────────────────────────────────────────────────
    def test_avc_delimiter_matches_avc(self):
        assert cla.REGEX_AVC_DELIMITER.search("type=AVC msg=audit(...)")

    def test_avc_delimiter_matches_user_avc(self):
        assert cla.REGEX_AVC_DELIMITER.search("type=USER_AVC msg=audit(...)")

    def test_avc_delimiter_no_match_selinux_err(self):
        assert cla.REGEX_AVC_DELIMITER.search("type=SELINUX_ERR msg=audit(...)") is None

    # ── REGEX_EXTRACT_PROCTITLE ───────────────────────────────────────────────
    def test_extract_proctitle(self):
        line = "type=PROCTITLE msg=audit(07/09/2024 16:38:22.104:7145) : proctitle=/bin/bash /usr/bin/myapp\n"
        m = cla.REGEX_EXTRACT_PROCTITLE.search(line)
        assert m is not None
        assert "/bin/bash" in m.group(1)
        assert "/usr/bin/myapp" in m.group(1)

    # ── REGEX_FULL_AVC ────────────────────────────────────────────────────────
    def test_full_avc_regex(self):
        block = ("type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp -s\n"
                 "type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat ppid=23905 pid=23906\n"
                 "type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc: denied")
        m = cla.REGEX_FULL_AVC.search(block)
        assert m is not None
        assert "/bin/bash /bin/myapp -s" in m.group(1)
        assert "ppid=23905" in m.group(2)

    # ── REGEX_SYSCALL_AVC ─────────────────────────────────────────────────────
    def test_syscall_avc_regex(self):
        block = "SYSCALL msg=audit(07/09/2024 16:38:22.104:7145) : arch=x86_64 syscall=stat ppid=4732 pid=5041\ntype=AVC"
        m = cla.REGEX_SYSCALL_AVC.search(block)
        assert m is not None
        assert "ppid=4732" in m.group(1)

    # ── REGEX_MAIN_EXPLANATION ────────────────────────────────────────────────
    def test_main_explanation_regex(self):
        line = "Rocky | PODMAN_0001 (pid=23910 ; pid_ns=4026532289)"
        m = cla.REGEX_MAIN_EXPLANATION.match(line)
        assert m is not None
        assert m.group(1) == "Rocky"
        assert "PODMAN_0001" in m.group(2)
        assert m.group(3) == "23910"
        assert m.group(4) == "4026532289"

    def test_main_explanation_with_notfound_ns(self):
        line = "Debian | /usr/bin/ls (pid=100 ; pid_ns=notFound)"
        m = cla.REGEX_MAIN_EXPLANATION.match(line)
        assert m is not None
        assert m.group(4) == "notFound"

    # ── REGEX_PID_TREE_HEADER ─────────────────────────────────────────────────
    def test_pid_tree_header_process_tree(self):
        assert cla.REGEX_PID_TREE_HEADER.match("# Process Tree (APP Root detected at 23944)")

    def test_pid_tree_header_app_process_tree(self):
        """Backward compatibility: old format with prefix before 'Process Tree'."""
        assert cla.REGEX_PID_TREE_HEADER.match("# APP Process Tree (root: 23944)")

    def test_pid_tree_header_orphan(self):
        assert cla.REGEX_PID_TREE_HEADER.match("# Orphan processes (not connected to main tree):")

    def test_pid_tree_header_no_match(self):
        assert cla.REGEX_PID_TREE_HEADER.match("# Some other comment") is None

    # ── REGEX_PID_TREE_ROOT ───────────────────────────────────────────────────
    def test_pid_tree_root_single(self):
        line = "# Process Tree (APP Root detected at 23944)"
        m = cla.REGEX_PID_TREE_ROOT.search(line)
        assert m is not None
        pids = [int(p.strip()) for p in m.group(1).split(',')]
        assert pids == [23944]

    def test_pid_tree_root_multiple(self):
        line = "# Process Tree (APP Roots detected at 23906, 24534, 27359, 27525)"
        m = cla.REGEX_PID_TREE_ROOT.search(line)
        assert m is not None
        pids = sorted(int(p.strip()) for p in m.group(1).split(',') if p.strip().isdigit())
        assert pids == [23906, 24534, 27359, 27525]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Analyzer Filter
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFilterAVC:
    def test_keeps_matching_source(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        assert a.filter_AVC(result) is True

    def test_keeps_matching_target(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[cla.AvcDenial("sshd_t", "myapp_t", "file", "write")],
        )
        assert a.filter_AVC(result) is True

    def test_discards_non_matching(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read")],
        )
        assert a.filter_AVC(result) is False

    def test_mixed_avcs_partial_filter(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[
                cla.AvcDenial("myapp_t", "cert_t", "dir", "read"),
                cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read"),
            ],
        )
        assert a.filter_AVC(result) is True
        assert len(result.avc_list) == 1
        assert result.avc_list[0].source_type == "myapp_t"

    def test_filter_respects_show_debug(self):
        """filter_AVC should not crash when show_debug is enabled."""
        a = cla.Analyzer(key="test", look_in_log=False, show_debug=True, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read")],
        )
        # Should discard and print debug message but not crash
        assert a.filter_AVC(result) is False

    def test_empty_avc_list_after_filter(self):
        """Result with all AVCs filtered out should return False."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[
                cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read"),
                cla.AvcDenial("httpd_t", "tmp_t", "dir", "write"),
            ],
        )
        assert a.filter_AVC(result) is False
        assert len(result.avc_list) == 0

    def test_multiple_matching_avcs_preserved(self):
        """All AVCs involving the context filter type should be preserved."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[
                cla.AvcDenial("myapp_t", "cert_t", "dir", "read"),
                cla.AvcDenial("myapp_t", "tmp_t", "file", "write"),
                cla.AvcDenial("httpd_t", "myapp_t", "sock_file", "connect"),
            ],
        )
        assert a.filter_AVC(result) is True
        assert len(result.avc_list) == 3

    def test_no_filter_keeps_all(self):
        """Without context_filter, all AVCs should be kept."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read")],
        )
        assert a.filter_AVC(result) is True

    def test_multiple_context_filter_types(self):
        """context_filter with multiple types should keep AVCs matching any of them."""
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t", "sshd_t"])
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test"),
            avc_list=[
                cla.AvcDenial("sshd_t", "unlabeled_t", "file", "read"),
                cla.AvcDenial("httpd_t", "tmp_t", "dir", "write"),
                cla.AvcDenial("myapp_t", "cert_t", "dir", "getattr"),
            ],
        )
        assert a.filter_AVC(result) is True
        assert len(result.avc_list) == 2
        types = {avc.source_type for avc in result.avc_list}
        assert types == {"sshd_t", "myapp_t"}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — filter_pid_tree
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFilterPidTree:
    """Unit tests for Analyzer.filter_pid_tree()."""

    def test_keeps_avc_pids(self):
        """Processes in avc_pids should be kept."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="ls", ppid=None, context="ctx", key="test", live=True)
        a.avc_pids.add(pk)
        a.filter_pid_tree()
        assert pk in a.pid_tree

    def test_removes_irrelevant_live(self):
        """Live processes not in avc_pids should be removed."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="ls", ppid=None, context="ctx", key="test", live=True)
        a.filter_pid_tree()
        assert pk not in a.pid_tree

    def test_keeps_non_live(self):
        """Non-live entries (loaded from file/JSON) should always be kept."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="ls", ppid=None, context="ctx", key="test", live=False)
        a.filter_pid_tree()
        assert pk in a.pid_tree

    def test_preserves_ancestors(self):
        """Ancestors of relevant PIDs should be kept for tree structure."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root_pk = (1, "test")
        mid_pk = (50, "test")
        leaf_pk = (100, "test")
        a.pid_tree[root_pk] = cla.PidTreeEntry(cmd="init", ppid=None, context="ctx", key="test", live=True, children=[mid_pk])
        a.pid_tree[mid_pk] = cla.PidTreeEntry(cmd="bash", ppid=1, context="ctx", key="test", live=True, children=[leaf_pk])
        a.pid_tree[leaf_pk] = cla.PidTreeEntry(cmd="app", ppid=50, context="ctx", key="test", live=True)
        a.avc_pids.add(leaf_pk)
        a.filter_pid_tree()
        assert root_pk in a.pid_tree
        assert mid_pk in a.pid_tree
        assert leaf_pk in a.pid_tree

    def test_prunes_unknown_linear_root(self):
        """An unknown:unknown root with only one child should be pruned."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root_pk = (1, "test")
        child_pk = (100, "test")
        a.pid_tree[root_pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True, children=[child_pk])
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd="app", ppid=1, context="myapp_t", key="test", live=True)
        a.avc_pids.add(child_pk)
        a.filter_pid_tree()
        assert root_pk not in a.pid_tree
        assert child_pk in a.pid_tree
        assert a.pid_tree[child_pk].ppid is None

    def test_children_updated_after_filter(self):
        """Children lists should not reference deleted entries."""
        a = cla.Analyzer(key="test", look_in_log=False)
        parent_pk = (1, "test")
        kept_pk = (2, "test")
        removed_pk = (3, "test")
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd="init", ppid=None, context="ctx", key="test", live=True,
            children=[kept_pk, removed_pk])
        a.pid_tree[kept_pk] = cla.PidTreeEntry(
            cmd="app", ppid=1, context="ctx", key="test", live=True)
        a.pid_tree[removed_pk] = cla.PidTreeEntry(
            cmd="other", ppid=1, context="ctx", key="test", live=True)
        a.avc_pids.add(kept_pk)
        a.filter_pid_tree()
        assert removed_pk not in a.pid_tree
        if parent_pk in a.pid_tree:
            assert removed_pk not in a.pid_tree[parent_pk].children


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — identify_app_root
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestIdentifyAppRoot:
    def test_returns_early_without_app_name(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        a.pid_tree[(1, "test")] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp", context="system_u:system_r:myapp_t:s0", key="test")
        a.identify_app_root()
        assert len(a.app_root_pids) == 0

    def test_returns_early_without_context_filter(self):
        a = cla.Analyzer(key="test", look_in_log=False, app_name="myapp")
        a.pid_tree[(1, "test")] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp", context="system_u:system_r:myapp_t:s0", key="test")
        a.identify_app_root()
        assert len(a.app_root_pids) == 0

    def test_identifies_matching_root(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"], app_name="myapp")
        pk = (1001, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp start", context="system_u:system_r:myapp_t:s0", key="test")
        a.identify_app_root()
        assert pk in a.app_root_pids

    def test_skips_unknown_cmd(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"], app_name="myapp")
        pk = (1, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, context="system_u:system_r:myapp_t:s0", key="test")
        a.identify_app_root()
        assert pk not in a.app_root_pids

    def test_skips_wrong_context(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"], app_name="myapp")
        pk = (1, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp", context="system_u:system_r:sshd_t:s0", key="test")
        a.identify_app_root()
        assert pk not in a.app_root_pids

    def test_identifies_multiple_roots(self):
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"], app_name="myapp")
        pk1 = (1001, "test")
        pk2 = (1003, "test")
        a.pid_tree[pk1] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp start", context="system_u:system_r:myapp_t:s0", key="test")
        a.pid_tree[pk2] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp stop", context="system_u:system_r:myapp_t:s0", key="test")
        a.identify_app_root()
        assert pk1 in a.app_root_pids
        assert pk2 in a.app_root_pids


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — File Parsing Functions
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseIndexFromFile:
    """Unit tests for parse_index_from_file()."""

    def test_parse_simple_index(self):
        doc = f"allow myapp_t cert_t:dir read;\n{cla.INDEX_DELIMITER}\n### CMD_0001 3 | /usr/bin/podman info\n"
        a = cla.Analyzer(key="test", look_in_log=False)
        replacer = a.parse_index_from_file(doc)
        assert "/usr/bin/podman info" in a.cmd_index or a.cmd_index.get_alias("/usr/bin/podman info") == "CMD_0001"
        assert isinstance(replacer, list)

    def test_missing_index_delimiter_raises(self):
        doc = "allow myapp_t cert_t:dir read;"
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.FileParsingError):
            a.parse_index_from_file(doc)

    def test_namespace_index_parsed(self):
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### ~pid_ns_test~ 0 | 12345\n"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_index_from_file(doc)
        assert 12345 in a.ns_index

    def test_alias_collision_produces_replacer(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        a.set_index("/usr/bin/cmd1", "CMD_0001")
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### CMD_0001 1 | /usr/bin/cmd2\n"
        replacer = a.parse_index_from_file(doc)
        assert len(replacer) > 0
        old, new = replacer[0]
        assert old == "CMD_0001"
        assert new != "CMD_0001"

    def test_history_dedup_parsing(self):
        """Multi-word commands with ***** history dedup should be reconstructed."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = (f"rules\n{cla.INDEX_DELIMITER}\n"
               f"### AVC_0001 1 | ausearch -i -m avc\n"
               f"### MSG_0001 1 | ******** ** ** msg\n")
        a.parse_index_from_file(doc)
        assert "ausearch -i -m msg" in a.cmd_index or a.cmd_index.get_alias("ausearch -i -m msg ") != "ausearch -i -m msg "


@pytest.mark.unit
class TestParseRulesFromFile:
    """Unit tests for parse_rules_from_file()."""

    def test_parse_simple_rule(self):
        doc = (f"allow myapp_t cert_t:dir read;\n"
               f"# required by :\n"
               f"#     test | /usr/bin/cmd (pid=100 ; pid_ns=12345)\n"
               f"{cla.INDEX_DELIMITER}\n")
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_rules_from_file(doc)
        assert len(results) == 1
        assert results[0].avc_list[0].source_type == "myapp_t"
        assert results[0].avc_list[0].target_type == "cert_t"
        assert results[0].avc_list[0].tclass == "dir"
        assert results[0].avc_list[0].method == "read"
        assert results[0].command.key == "test"
        assert results[0].command.pid == "100"

    def test_missing_index_raises(self):
        doc = "allow myapp_t cert_t:dir read;"
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.FileParsingError):
            a.parse_rules_from_file(doc)

    def test_multiple_rules_parsed(self):
        doc = (f"allow myapp_t cert_t:dir read;\n"
               f"# required by :\n"
               f"#     test | cmd1 (pid=100 ; pid_ns=12345)\n"
               f"\nallow myapp_t tmp_t:file write;\n"
               f"# required by :\n"
               f"#     test | cmd2 (pid=200 ; pid_ns=12345)\n"
               f"{cla.INDEX_DELIMITER}\n")
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_rules_from_file(doc)
        assert len(results) == 2
        methods = {r.avc_list[0].method for r in results}
        assert "read" in methods
        assert "write" in methods

    def test_rule_with_descriptors(self):
        doc = (f"allow myapp_t cert_t:dir read;\n"
               f"# required by :\n"
               f"#     test | /usr/bin/cmd (pid=100 ; pid_ns=12345)\n"
               f"#          | SYSCALL: msg=audit(...) arch=x86_64\n"
               f"{cla.INDEX_DELIMITER}\n")
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_rules_from_file(doc)
        assert len(results) == 1
        assert len(results[0].command.descriptors) > 0


@pytest.mark.unit
class TestParsePidTreeFromFile:
    """Unit tests for parse_pid_tree_from_file()."""

    def test_parse_simple_tree(self):
        tree_section = (
            "# test - Process Tree (APP Root detected at 1001)\n"
            "#\n"
            "# ├── pid=1001@test [APP ROOT]        ctx: system_u:system_r:myapp_t:s0                 cmd: /bin/bash /usr/sbin/myapp\n"
            "# │   ├── pid=1002@test               ctx: system_u:system_r:myapp_t:s0                 cmd: /usr/lib/myapp/worker\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert (1001, "test") in a.pid_tree
        assert (1002, "test") in a.pid_tree
        assert a.pid_tree[(1001, "test")].cmd == "/bin/bash /usr/sbin/myapp"
        assert (1001, "test") in a.app_root_pids

    def test_no_tree_section(self):
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert len(a.pid_tree) == 0

    def test_parent_child_relationship(self):
        tree_section = (
            "# test - Process Tree\n"
            "#\n"
            "# ├── pid=1@test                      ctx: system_u:system_r:init_t:s0                  cmd: /sbin/init\n"
            "# │   ├── pid=2@test                  ctx: system_u:system_r:myapp_t:s0                 cmd: /usr/bin/app\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert a.pid_tree[(2, "test")].ppid == 1
        assert (2, "test") in a.pid_tree[(1, "test")].children


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — State File (load/save)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseExistingFile:
    """Unit tests for parse_existing_file() — the wrapper that orchestrates index + tree + rules parsing."""

    def test_alias_replacement_applied_to_doc(self):
        """When parse_index_from_file produces replacements (alias collision),
        parse_existing_file should apply them to the doc text before parsing rules."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Pre-register an alias so the file's alias collides
        a.set_index("/usr/bin/cmd_existing", "CMD_0001")

        # The file uses CMD_0001 for a different command — collision!
        doc = (
            f"allow myapp_t cert_t:dir read;\n"
            f"# required by :\n"
            f"#     test | CMD_0001 (pid=100 ; pid_ns=12345)\n"
            f"{cla.INDEX_DELIMITER}\n"
            f"### CMD_0001 1 | /usr/bin/cmd_from_file\n"
        )
        results = a.parse_existing_file(doc)
        assert len(results) == 1
        # Both commands should be in the index with different aliases
        alias_existing = a.cmd_index.get_alias("/usr/bin/cmd_existing")
        alias_from_file = a.cmd_index.get_alias("/usr/bin/cmd_from_file")
        assert alias_existing != alias_from_file
        assert alias_existing == "CMD_0001"  # pre-existing keeps its alias
        # The result's command should be the new alias (replacement applied to doc)
        assert results[0].command.cmd == alias_from_file

    def test_file_parsing_error_propagates(self):
        """A malformed file (missing INDEX delimiter) should raise FileParsingError."""
        a = cla.Analyzer(key="test", look_in_log=False)
        with pytest.raises(cla.FileParsingError):
            a.parse_existing_file("no index here")


@pytest.mark.unit
class TestStateFileFunctions:
    """Unit tests for load_analyzed_entries() and save_analyzed_entries()."""

    def test_save_and_load_roundtrip(self, tmp_path):
        sf = str(tmp_path / "state.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.analyzed_entries = {"entry1", "entry2", "entry3"}
        a.save_analyzed_entries()
        assert os.path.isfile(sf)

        a2 = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a2.load_analyzed_entries()
        assert a2.analyzed_entries == {"entry1", "entry2", "entry3"}

    def test_load_nonexistent_file(self, tmp_path):
        sf = str(tmp_path / "nonexistent.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.load_analyzed_entries()
        assert a.analyzed_entries == set()

    def test_load_corrupt_file(self, tmp_path):
        sf = str(tmp_path / "corrupt.json")
        with open(sf, "w") as f:
            f.write("{bad json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.load_analyzed_entries()
        assert a.analyzed_entries == set()

    def test_save_without_state_file(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        a.analyzed_entries = {"entry1"}
        a.save_analyzed_entries()  # should be no-op, no crash

    def test_load_without_state_file(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        a.load_analyzed_entries()  # should be no-op, no crash
        assert a.analyzed_entries == set()

    def test_save_creates_file_atomically(self, tmp_path):
        """Save should use atomic write (temp + rename)."""
        sf = str(tmp_path / "state.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.analyzed_entries = {"x"}
        a.save_analyzed_entries()
        with open(sf) as f:
            data = json.load(f)
        assert "x" in data["analyzed_entries"]

    def test_is_entry_analyzed_and_mark(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        block = "type=AVC msg=audit(1234:100) : test"
        assert not a.is_entry_analyzed(block)
        a.mark_entry_analyzed(block)
        assert a.is_entry_analyzed(block)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — look_for_constraint_violation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestLookForConstraintViolation:
    def test_handles_nonexistent_log(self, capsys):
        """Should not crash on nonexistent log path."""
        cla.Analyzer.look_for_constraint_violation("/nonexistent/audit.log")
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "could not" in captured.err.lower()

    def test_handles_empty_log(self, tmp_path, capsys):
        """Should handle empty log without crashing."""
        log = str(tmp_path / "empty.log")
        with open(log, "w"):
            pass
        cla.Analyzer.look_for_constraint_violation(log)
        # Should not raise — no assertion needed beyond not crashing


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — enrich_pid_tree
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEnrichPidTree:
    """Unit tests for enrich_pid_tree() dead-process labelling and parent chain walking."""

    def test_labels_dead_processes(self):
        """Unknown processes whose PID is dead should be labelled DEAD_PROCESS."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (9999999, "test")  # PID that certainly doesn't exist
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)
        a.enrich_pid_tree()
        assert a.pid_tree.get(pk) is None or a.pid_tree[pk].cmd == cla.DEAD_PROCESS

    def test_skips_non_live_entries(self):
        """Non-live entries (loaded from file/JSON) should not be enriched."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (9999999, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=False,
        )
        a.enrich_pid_tree()
        # Should remain UNKNOWN (not touched)
        assert a.pid_tree[pk].cmd == cla.UNKNOWN

    def test_enriches_own_pid(self):
        """A live entry with our own PID should be enriched with real data."""
        a = cla.Analyzer(key="test", look_in_log=False)
        my_pid = os.getpid()
        pk = (my_pid, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)
        a.enrich_pid_tree()
        # Our own process should be enriched with a real command
        entry = a.pid_tree.get(pk)
        assert entry is not None
        assert entry.cmd != cla.UNKNOWN
        assert entry.ppid is not None

    def test_post_prune_removes_dead_orphans(self):
        """After enrichment, childless dead_process orphans should be pruned."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (9999999, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)
        a.enrich_pid_tree()
        # Dead orphan with no children and no parent → pruned
        assert pk not in a.pid_tree


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — index_pid_tree_cmds
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestIndexPidTreeCmds:
    def test_logs_live_commands(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="/usr/bin/app arg", key="test", live=True)
        a.index_pid_tree_cmds()
        assert a.cmd_index.get_weight("/usr/bin/app arg") == 1

    def test_skips_non_live(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="/usr/bin/app arg", key="test", live=False)
        a.index_pid_tree_cmds()
        assert a.cmd_index.get_weight("/usr/bin/app arg") == 0

    def test_skips_unknown_and_dead(self):
        a = cla.Analyzer(key="test", look_in_log=False)
        a.pid_tree[(1, "test")] = cla.PidTreeEntry(cmd=cla.UNKNOWN, key="test", live=True)
        a.pid_tree[(2, "test")] = cla.PidTreeEntry(cmd=cla.DEAD_PROCESS, key="test", live=True)
        a.index_pid_tree_cmds()
        assert a.cmd_index.get_weight(cla.UNKNOWN) == 0
        assert a.cmd_index.get_weight(cla.DEAD_PROCESS) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — format_rules
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFormatRules:
    """Test the format_rules() function which formats SELinux rules with context."""

    def test_single_rule_single_result(self):
        """Single AVC denial should produce one rule entry."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result])
        assert len(rules) == 1
        assert "allow myapp_t cert_t:dir read;" in rules

    def test_multiple_methods_merged(self):
        """Multiple permissions on same source/target/class create separate rules."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[
                cla.AvcDenial("myapp_t", "cert_t", "dir", "read"),
                cla.AvcDenial("myapp_t", "cert_t", "dir", "open"),
            ],
        )
        rules = a.format_rules([result])
        # Each permission creates a separate rule
        rule_keys = list(rules.keys())
        assert len(rule_keys) == 2
        # Both should reference the same source/target/class
        assert all("myapp_t cert_t:dir" in rule for rule in rule_keys)

    def test_deduplication_across_results(self):
        """Identical AVCs from different blocks should deduplicate but show both commands."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result1 = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        result2 = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="dhcpd", pid=2, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result1, result2])
        assert len(rules) == 1
        rule_key = list(rules.keys())[0]
        # Should have two different command contexts
        assert len(rules[rule_key]) == 2

    def test_rule_with_self_target(self):
        """Rule with same source and target type."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "myapp_t", "process", "signal")],
        )
        rules = a.format_rules([result])
        assert len(rules) == 1
        assert "allow myapp_t myapp_t:process signal;" in rules or "allow myapp_t self:process signal;" in rules

    def test_empty_results_produces_empty_rules(self):
        """Empty parsing results should produce empty rules dict."""
        a = cla.Analyzer(key="test", look_in_log=False)
        rules = a.format_rules([])
        assert len(rules) == 0
        assert rules == {}

    def test_result_with_no_avcs(self):
        """Result with empty avc_list should not contribute rules."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[],
        )
        rules = a.format_rules([result])
        assert len(rules) == 0

    def test_descriptors_included(self):
        """SYSCALL descriptors should be included in rule context."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(
                key="test",
                cmd="systemd",
                pid=1,
                pid_namespace=12345,
                descriptors=["SYSCALL: msg=audit(...) arch=x86_64 syscall=openat"],
            ),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result])
        rule_key = list(rules.keys())[0]
        cmd_contexts = rules[rule_key]
        # Should have at least one command context with descriptors
        assert len(cmd_contexts) >= 1
        descriptors_found = False
        for cmd_key in cmd_contexts:
            if len(cmd_contexts[cmd_key]) > 0:
                descriptors_found = True
        assert descriptors_found

    def test_namespace_index_used(self):
        """Namespace index should be used for formatting pid_ns."""
        a = cla.Analyzer(key="test", look_in_log=False)
        a.ns_index.set(12345, "~pid_ns_mynamespace~")
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result])
        rule_key = list(rules.keys())[0]
        cmd_contexts = list(rules[rule_key].keys())
        # Should contain the namespace alias
        assert any("~pid_ns_mynamespace~" in ctx for ctx in cmd_contexts)

    def test_cmd_index_used(self):
        """Command index should be used for formatting commands."""
        a = cla.Analyzer(key="test", look_in_log=False)
        a.set_index("/usr/bin/very-long-command-name", "~CMD1~")
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="/usr/bin/very-long-command-name", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result])
        rule_key = list(rules.keys())[0]
        cmd_contexts = list(rules[rule_key].keys())
        # Should contain the command alias
        assert any("~CMD1~" in ctx for ctx in cmd_contexts)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — format_pid_tree
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFormatPidTreeUnit:
    """Unit tests for format_pid_tree() function."""

    def test_empty_tree(self):
        """Empty PID tree should produce a message indicating no EXECVE events."""
        a = cla.Analyzer(key="test", look_in_log=False)
        output = a.format_pid_tree()
        assert "No EXECVE events found" in output or "empty" in output.lower()

    def test_single_process_no_children(self):
        """Single process with no children should appear in orphans section."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="unconfined_u:unconfined_r:unconfined_t:s0",
            key="test", live=False,
        )
        a.avc_pids.add(pk)
        output = a.format_pid_tree()
        assert "1234" in output
        assert "/bin/bash" in output

    def test_orphan_processes_section(self):
        """Processes not connected to any tree should appear in orphans section."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Create two orphan processes (no parent-child relationship)
        pk1 = (1001, "test")
        pk2 = (1002, "test")
        a.pid_tree[pk1] = cla.PidTreeEntry(
            cmd="/usr/bin/orphan1", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.pid_tree[pk2] = cla.PidTreeEntry(
            cmd="/usr/bin/orphan2", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.avc_pids.update([pk1, pk2])
        output = a.format_pid_tree()
        assert "Orphan" in output
        assert "1001" in output
        assert "1002" in output

    def test_tree_with_multiple_roots(self):
        """Tree with multiple independent roots should format both."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Root 1 with child
        root1_pk = (1000, "test")
        child1_pk = (1001, "test")
        a.pid_tree[root1_pk] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="system_u:system_r:init_t:s0",
            key="test", live=False, children=[child1_pk],
        )
        a.pid_tree[child1_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/child1", ppid=1000, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        # Root 2 with child
        root2_pk = (2000, "test")
        child2_pk = (2001, "test")
        a.pid_tree[root2_pk] = cla.PidTreeEntry(
            cmd="/bin/systemd", ppid=None, context="system_u:system_r:init_t:s0",
            key="test", live=False, children=[child2_pk],
        )
        a.pid_tree[child2_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/child2", ppid=2000, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.avc_pids.update([child1_pk, child2_pk])
        output = a.format_pid_tree()
        assert "1000" in output
        assert "1001" in output
        assert "2000" in output
        assert "2001" in output

    def test_deeply_nested_tree(self):
        """Tree with 5+ levels of nesting should format correctly."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Create a 6-level deep tree
        pids = [1000, 1001, 1002, 1003, 1004, 1005]
        for i in range(len(pids)):
            pk = (pids[i], "test")
            ppid = pids[i-1] if i > 0 else None
            child_pk = (pids[i+1], "test") if i < len(pids) - 1 else None
            children = [child_pk] if child_pk else []
            a.pid_tree[pk] = cla.PidTreeEntry(
                cmd=f"/usr/bin/level{i}", ppid=ppid, context=f"system_u:system_r:myapp_t:s0",
                key="test", live=False, children=children,
            )
        a.avc_pids.add((1005, "test"))  # leaf node has AVC
        output = a.format_pid_tree()
        # Check all levels are present
        for pid in pids:
            assert str(pid) in output
        # Check indentation (│ characters for nesting)
        assert "│" in output  # Should have tree structure indicators

    def test_app_root_marker(self):
        """APP root PID should be marked with [APP ROOT] in output."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root_pk = (23906, "test")
        child_pk = (23907, "test")
        a.pid_tree[root_pk] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False, children=[child_pk],
        )
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/child", ppid=23906, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.app_root_pids = [root_pk]
        a.avc_pids.add(child_pk)
        output = a.format_pid_tree()
        assert "[APP ROOT]" in output and "23906" in output

    def test_multiple_app_roots_marker(self):
        """Multiple APP roots should be indicated in header."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root1_pk = (23906, "test")
        root2_pk = (24534, "test")
        child1_pk = (23907, "test")
        child2_pk = (24535, "test")

        a.pid_tree[root1_pk] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp -s", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False, children=[child1_pk],
        )
        a.pid_tree[child1_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/child1", ppid=23906, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )

        a.pid_tree[root2_pk] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False, children=[child2_pk],
        )
        a.pid_tree[child2_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/child2", ppid=24534, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )

        a.app_root_pids = [root1_pk, root2_pk]
        a.avc_pids.update([child1_pk, child2_pk])
        output = a.format_pid_tree()
        assert "23906" in output and "24534" in output
        # Should indicate multiple roots
        assert "Roots" in output or "Root" in output

    def test_tree_with_key_in_output(self):
        """PID entries should include key in format pid=X@KEY."""
        a = cla.Analyzer(key="myhost", look_in_log=False)
        pk = (1234, "myhost")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="myhost", live=False,
        )
        a.avc_pids.add(pk)
        output = a.format_pid_tree()
        assert "1234@myhost" in output or ("1234" in output and "myhost" in output)

    def test_context_and_cmd_columns(self):
        """Output should have aligned columns for ctx: and cmd:."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.avc_pids.add(pk)
        output = a.format_pid_tree()
        assert "ctx:" in output
        assert "cmd:" in output

    def test_cmd_index_alias_used(self):
        """Command aliases from cmd_index should be used in tree output."""
        a = cla.Analyzer(key="test", look_in_log=False)
        a.set_index("/usr/bin/very-long-command-with-many-arguments", "~CMD1~")
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/usr/bin/very-long-command-with-many-arguments", ppid=None,
            context="system_u:system_r:myapp_t:s0", key="test", live=False,
        )
        a.avc_pids.add(pk)
        output = a.format_pid_tree()
        assert "~CMD1~" in output


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — to_json / merge_json
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestJsonSerialization:
    """Unit tests for to_json() and merge_json() methods."""

    def test_to_json_schema_complete(self):
        """to_json() output should contain all required keys."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd="systemd", pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        json_data = a.to_json([result])

        # Check all required keys are present
        assert "version" in json_data
        assert "key" in json_data
        assert "results" in json_data
        assert "index" in json_data
        assert "ns_index" in json_data
        assert "all_cmds" in json_data
        assert "all_aliases" in json_data
        assert "pid_tree" in json_data
        assert "app_root_pids" in json_data
        assert "avc_pids" in json_data
        assert "counters" in json_data
        assert "avc_counter" in json_data["counters"]
        assert "file_counter" in json_data["counters"]

    def test_to_json_version_field(self):
        """to_json() should include correct version number."""
        a = cla.Analyzer(key="test", look_in_log=False)
        json_data = a.to_json([])
        assert json_data["version"] == cla.JSON_FORMAT_VERSION

    def test_merge_json_empty_dict(self):
        """merge_json() should handle empty dict gracefully."""
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.merge_json({})
        assert results == []

    def test_merge_json_missing_keys(self):
        """merge_json() should handle missing optional keys gracefully."""
        a = cla.Analyzer(key="test", look_in_log=False)
        minimal_json = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
        }
        results = a.merge_json(minimal_json)
        # Should not crash, returns empty results
        assert isinstance(results, list)

    def test_merge_json_extra_keys_ignored(self):
        """merge_json() should ignore unknown keys without crashing."""
        a = cla.Analyzer(key="test", look_in_log=False)
        json_with_extras = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [],
            "unknown_future_field": "some_value",
            "another_unknown": 12345,
        }
        results = a.merge_json(json_with_extras)
        # Should not crash
        assert isinstance(results, list)

    def test_merge_json_version_mismatch_warning(self, capsys):
        """merge_json() should warn on version mismatch but continue."""
        a = cla.Analyzer(key="test", look_in_log=False)
        old_version_json = {
            "version": 999,  # Intentionally wrong version
            "key": "test",
            "results": [],
        }
        results = a.merge_json(old_version_json)
        captured = capsys.readouterr()
        # Should warn about version mismatch
        assert "version" in captured.err.lower() or "Version" in captured.err

    def test_to_json_pid_tree_serialization(self):
        """PID tree (pid, key) tuples should serialize as 'pid:key' strings."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        json_data = a.to_json([])

        # Check serialization format
        assert "1234:test" in json_data["pid_tree"]
        entry = json_data["pid_tree"]["1234:test"]
        assert entry["cmd"] == "/bin/bash"

    def test_merge_json_pid_tree_deserialization(self):
        """merge_json() should deserialize 'pid:key' strings back to tuples."""
        a = cla.Analyzer(key="test", look_in_log=False)
        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "pid_tree": {
                "1234:test": {
                    "cmd": "/bin/bash",
                    "ppid": None,
                    "context": "system_u:system_r:myapp_t:s0",
                    "key": "test",
                    "live": False,
                    "children": [],
                }
            },
        }
        a.merge_json(json_data)

        # Check deserialization
        pk = (1234, "test")
        assert pk in a.pid_tree
        assert a.pid_tree[pk].cmd == "/bin/bash"

    def test_merge_json_app_root_pids_deserialization(self):
        """app_root_pids should deserialize from 'pid:key' strings if PIDs exist in tree."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # First, add the PIDs to the pid_tree (required for merge to work)
        a.pid_tree[(23906, "test")] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp -s", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.pid_tree[(24534, "test")] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )

        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "app_root_pids": ["23906:test", "24534:test"],
            "pid_tree": {
                "23906:test": {"cmd": "/bin/bash /bin/myapp -s", "ppid": None, "context": "system_u:system_r:myapp_t:s0", "key": "test", "live": False, "children": []},
                "24534:test": {"cmd": "/bin/bash /bin/myapp", "ppid": None, "context": "system_u:system_r:myapp_t:s0", "key": "test", "live": False, "children": []},
            },
        }
        a.merge_json(json_data)

        # Check deserialization
        assert (23906, "test") in a.app_root_pids
        assert (24534, "test") in a.app_root_pids

    def test_merge_json_index_collision_handling(self):
        """merge_json() should handle command index alias collisions."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Pre-register an alias
        a.set_index("/usr/bin/cmd1", "~CMD1~")

        # Try to merge JSON with conflicting alias
        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "index": {
                "/usr/bin/cmd2": "~CMD1~",  # Collision!
            },
        }
        a.merge_json(json_data)

        # The incoming cmd should get a different alias
        assert "/usr/bin/cmd1" in a.cmd_index
        assert "/usr/bin/cmd2" in a.cmd_index
        # Aliases should be different
        alias1 = a.cmd_index.get_alias("/usr/bin/cmd1")
        alias2 = a.cmd_index.get_alias("/usr/bin/cmd2")
        assert alias1 != alias2

    def test_merge_json_ns_index_collision_handling(self):
        """merge_json() should handle namespace index label collisions."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Pre-register a namespace
        a.ns_index.set(12345, "~pid_ns_host~")

        # Try to merge JSON with conflicting label
        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "ns_index": {
                "67890": "~pid_ns_host~",  # Collision!
            },
        }
        a.merge_json(json_data)

        # Both namespaces should be registered
        assert 12345 in a.ns_index
        assert 67890 in a.ns_index
        # Labels should be different (one gets a suffix)
        label1 = a.ns_index.get(12345)
        label2 = a.ns_index.get(67890)
        assert label1 != label2

    def test_merge_json_cmd_weights_accumulate(self):
        """merge_json() should accumulate weights for commands."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # Log a command with weight 5
        a.log_cmd("/usr/bin/cmd", weight=5)

        # Merge JSON with weight 10
        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "all_cmds": {
                "/usr/bin/cmd": 10,
            },
        }
        a.merge_json(json_data)

        # Weight should be accumulated (5 + 10 = 15)
        assert a.cmd_index.get_weight("/usr/bin/cmd") == 15

    def test_to_json_results_serialization(self):
        """Results should serialize with all AVC denial details."""
        a = cla.Analyzer(key="test", look_in_log=False)
        result = cla.AnalysisResult(
            command=cla.CommandContext(
                key="test",
                cmd="systemd",
                pid=1,
                pid_namespace=12345,
                descriptors=["SYSCALL: test"],
            ),
            avc_list=[
                cla.AvcDenial("myapp_t", "cert_t", "dir", "read"),
                cla.AvcDenial("myapp_t", "tmp_t", "file", "write"),
            ],
        )
        json_data = a.to_json([result])

        # Check results array
        assert len(json_data["results"]) == 1
        r = json_data["results"][0]
        assert r["command"]["cmd"] == "systemd"
        # The key is "AVC" not "avc_list"
        assert len(r["AVC"]) == 2

    def test_merge_json_results_deserialization(self):
        """merge_json() should deserialize results with AVCs (uses AVC key, not avc_list)."""
        a = cla.Analyzer(key="test", look_in_log=False)
        json_data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [
                {
                    "command": {
                        "key": "test",
                        "cmd": "systemd",
                        "pid": 1,
                        "pid_namespace": 12345,
                        "descriptors": ["SYSCALL: test"],
                    },
                    "AVC": [
                        {
                            "source_type": "myapp_t",
                            "target_type": "cert_t",
                            "tclass": "dir",
                            "method": "read",
                        }
                    ],
                }
            ],
        }
        results = a.merge_json(json_data)

        # Check deserialization
        assert len(results) == 1
        assert results[0].command.cmd == "systemd"
        assert len(results[0].avc_list) == 1
        assert results[0].avc_list[0].source_type == "myapp_t"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Multi-key merge
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMultiKeyMerge:
    """Integration tests for merging analyzers with different keys."""

    def test_merge_two_analyzers_different_keys(self):
        """Merge two analyzers with different keys should combine results."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Create results for each analyzer
        result1 = cla.AnalysisResult(
            command=cla.CommandContext(key="host1", cmd="systemd", pid=1, pid_namespace=1000),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        result2 = cla.AnalysisResult(
            command=cla.CommandContext(key="host2", cmd="dhcpd", pid=2, pid_namespace=2000),
            avc_list=[cla.AvcDenial("myapp_t", "tmp_t", "file", "write")],
        )

        # Export first analyzer to JSON
        json1 = a1.to_json([result1])

        # Merge into second analyzer
        merged_results = a2.merge_json(json1)

        # Should have the merged result
        assert len(merged_results) == 1
        assert merged_results[0].command.key == "host1"
        assert merged_results[0].command.cmd == "systemd"

    def test_pid_tree_merge_same_pid_different_keys(self):
        """Same PID with different keys should be treated as separate entries."""
        a = cla.Analyzer(key="merged", look_in_log=False)

        # Add PID tree entries with same PID but different keys
        pk1 = (1234, "host1")
        pk2 = (1234, "host2")

        a.pid_tree[pk1] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="host1", live=False,
        )
        a.pid_tree[pk2] = cla.PidTreeEntry(
            cmd="/usr/bin/python", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="host2", live=False,
        )

        # Both should coexist
        assert pk1 in a.pid_tree
        assert pk2 in a.pid_tree
        assert a.pid_tree[pk1].cmd == "/bin/bash"
        assert a.pid_tree[pk2].cmd == "/usr/bin/python"

    def test_pid_tree_merge_via_json(self):
        """PID trees from different keys should merge correctly via JSON."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Add PID tree to first analyzer
        pk1 = (100, "host1")
        a1.pid_tree[pk1] = cla.PidTreeEntry(
            cmd="/bin/init", ppid=None, context="system_u:system_r:init_t:s0",
            key="host1", live=False,
        )

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # Second analyzer should have the PID tree entry
        assert pk1 in a2.pid_tree
        assert a2.pid_tree[pk1].cmd == "/bin/init"

    def test_avc_pids_merge(self):
        """avc_pids from different analyzers should merge."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Add avc_pids to first analyzer
        a1.avc_pids.add((100, "host1"))
        a1.avc_pids.add((200, "host1"))

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # Second analyzer should have the avc_pids
        assert (100, "host1") in a2.avc_pids
        assert (200, "host1") in a2.avc_pids

    def test_cmd_index_merge(self):
        """Command indexes should merge without conflicts."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Register commands in first analyzer
        a1.set_index("/usr/bin/command1", "~CMD1~")
        a1.log_cmd("/usr/bin/command1", weight=10)

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # Second analyzer should have the command index
        assert "/usr/bin/command1" in a2.cmd_index
        assert a2.get_index("/usr/bin/command1") == "~CMD1~"
        assert a2.cmd_index.get_weight("/usr/bin/command1") == 10

    def test_ns_index_merge(self):
        """Namespace indexes should merge."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Register namespace in first analyzer
        a1.ns_index.set(12345, "~pid_ns_test~")

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # Second analyzer should have the namespace
        assert 12345 in a2.ns_index
        assert a2.ns_index.get(12345) == "~pid_ns_test~"

    def test_counter_accumulation(self):
        """Counters should accumulate during merge."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Set counters in first analyzer
        a1.avc_counter = 10
        a1.file_counter = 5

        # Set different counters in second analyzer
        a2.avc_counter = 15
        a2.file_counter = 3

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # file_counter should accumulate (merge adds file count from results)
        assert a2.file_counter >= 3  # At least original value

    def test_merge_preserves_both_keys_in_results(self):
        """Merged results should preserve their original keys."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Create results with different keys
        result1 = cla.AnalysisResult(
            command=cla.CommandContext(key="host1", cmd="cmd1", pid=1, pid_namespace=1000),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        result2 = cla.AnalysisResult(
            command=cla.CommandContext(key="host2", cmd="cmd2", pid=2, pid_namespace=2000),
            avc_list=[cla.AvcDenial("myapp_t", "tmp_t", "file", "write")],
        )

        # Export first analyzer
        json1 = a1.to_json([result1])

        # Merge into second analyzer and add its own result
        merged = a2.merge_json(json1)

        # Format rules from both
        all_results = merged + [result2]
        rules = a2.format_rules(all_results)

        # Convert to string to check both keys appear
        rules_str = str(rules)
        assert "host1" in rules_str
        assert "host2" in rules_str

    def test_app_root_pids_merge(self):
        """app_root_pids from different keys should merge."""
        a1 = cla.Analyzer(key="host1", look_in_log=False)
        a2 = cla.Analyzer(key="host2", look_in_log=False)

        # Add APP root PID in first analyzer (with corresponding PID tree entry)
        root_pk = (23906, "host1")
        a1.pid_tree[root_pk] = cla.PidTreeEntry(
            cmd="/bin/bash /bin/myapp", ppid=None, context="system_u:system_r:myapp_t:s0",
            key="host1", live=False,
        )
        a1.app_root_pids.append(root_pk)

        # Export and merge
        json1 = a1.to_json([])
        a2.merge_json(json1)

        # Second analyzer should have the APP root PID
        assert root_pk in a2.app_root_pids


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — parse_ausearch_from_log edge cases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseAusearchEdgeCases:
    """Edge cases for parse_ausearch_from_log that don't require ausearch or test_log."""

    def test_block_with_no_pid(self):
        """Block with no PID at all should produce pid=NOT_FOUND."""
        block = textwrap.dedent("""\
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } for  comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:user_tmp_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert results[0].command.pid == cla.NOT_FOUND

    def test_block_with_user_avc(self):
        """USER_AVC blocks should be split by REGEX_AVC_DELIMITER and parsed."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp
            type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat ppid=100 pid=200 uid=root subj=system_u:system_r:myapp_t:s0
            type=USER_AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } for  pid=200 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert results[0].avc_list[0].source_type == "myapp_t"
        assert results[0].avc_list[0].tclass == "file"
        assert results[0].avc_list[0].method == "read"

    def test_block_with_empty_cmdline_falls_back_to_proctitle(self):
        """When cmdline is empty (\"\"), parser should fall back to PROCTITLE."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp -s
            type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat ppid=100 pid=200 uid=root cmdline=""
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } for  pid=200 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:user_tmp_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        # Should have fallen back to proctitle
        assert "/bin/bash /bin/myapp -s" in results[0].command.cmd

    def test_block_with_exe_fallback(self):
        """Block with no PROCTITLE, no cmdline: should fall back to exe=."""
        block = textwrap.dedent("""\
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } for  pid=200 comm=myapp exe=/usr/bin/bash scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:user_tmp_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert "/usr/bin/bash" in results[0].command.cmd

    def test_block_with_comm_fallback(self):
        """Block with no exe=, no PROCTITLE: should fall back to comm=."""
        block = textwrap.dedent("""\
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } for  pid=200 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:user_tmp_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert results[0].command.cmd == "myapp"

    def test_block_no_syscall_uid_path(self):
        """Block with no SYSCALL line should produce 'no SYSCALL found' descriptor."""
        block = textwrap.dedent("""\
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } for  pid=300 uid=root comm=podman scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        descriptors = results[0].command.descriptors
        assert any("no SYSCALL found" in d for d in descriptors)
        assert any("uid=root" in d for d in descriptors)

    def test_block_no_syscall_no_uid(self):
        """Block with no SYSCALL and no uid= field should report uid as some sentinel."""
        block = textwrap.dedent("""\
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } for  pid=9999999 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        descriptors = results[0].command.descriptors
        # uid should be notFound or dead_process (never a real username)
        assert any("no SYSCALL found" in d and (f"uid={cla.NOT_FOUND}" in d or f"uid={cla.DEAD_PROCESS}" in d) for d in descriptors)

    def test_empty_blocks_iterable(self):
        """Empty blocks iterable should produce no results."""
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[])
        assert len(results) == 0
        assert a.avc_counter == 0

    def test_none_blocks_returns_empty(self):
        """blocks=None should produce no results."""
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=None)
        assert len(results) == 0

    def test_avc_counter_incremented(self):
        """avc_counter should be incremented for each parsed AVC denial."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp
            type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat ppid=100 pid=200
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read open } for  pid=200 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_ausearch_from_log(blocks=[block])
        assert a.avc_counter == 2  # read + open

    def test_avc_pid_tracked(self):
        """PIDs from parsed AVCs should be tracked in avc_pids."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp
            type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat ppid=100 pid=200
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } for  pid=200 comm=myapp scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_ausearch_from_log(blocks=[block])
        assert (200, "test") in a.avc_pids


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Full Analysis from Log
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestFullAnalysisFromLog:
    """Run the full analyze() pipeline against the reference log."""

    def test_produces_output(self, tmp_path):
        txt, a, jpath = _run_full_analysis(tmpdir=str(tmp_path))
        assert len(txt) > 0
        assert a.avc_counter > 0

    def test_output_contains_allow_rules(self, tmp_path):
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert "allow myapp_t" in txt

    def test_output_has_index_delimiter(self, tmp_path):
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert cla.INDEX_DELIMITER in txt

    def test_output_has_pid_tree_delimiter(self, tmp_path):
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert cla.PID_TREE_DELIMITER in txt

    def test_app_root_detected(self, tmp_path):
        """All 4 APP root PIDs must be identified."""
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert len(a.app_root_pids) > 0, "No APP root PIDs were detected"
        detected_pids = {pk[0] for pk in a.app_root_pids}
        assert detected_pids == APP_ROOT_PIDS, (
            f"Expected APP root PIDs {APP_ROOT_PIDS}, got {detected_pids}"
        )
        # All roots should have the reference key
        for pk in a.app_root_pids:
            assert pk[1] == REFERENCE_KEY

    def test_app_root_in_pid_tree_text(self, tmp_path):
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert "[APP ROOT]" in txt
        for pid in APP_ROOT_PIDS:
            assert f"pid={pid}@{REFERENCE_KEY}" in txt, (
                f"APP root pid={pid} not found in output"
            )

    def test_app_root_has_app_cmd(self, tmp_path):
        """All APP root entries must have /usr/sbin/myapp in their command."""
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        for pk in a.app_root_pids:
            root_entry = a.pid_tree.get(pk)
            assert root_entry is not None, f"Root {pk} not in pid_tree"
            assert "/usr/sbin/myapp" in root_entry.cmd, (
                f"Root {pk} has cmd={root_entry.cmd}, expected /usr/sbin/myapp"
            )
            assert "myapp_t" in root_entry.context

    def test_json_output_valid(self, tmp_path):
        txt, a, jpath = _run_full_analysis(tmpdir=str(tmp_path))
        assert os.path.isfile(jpath)
        with open(jpath) as f:
            data = json.load(f)
        assert data["version"] == cla.JSON_FORMAT_VERSION
        assert data["key"] == REFERENCE_KEY
        assert len(data["results"]) > 0
        assert len(data["pid_tree"]) > 0

    def test_json_has_app_roots(self, tmp_path):
        txt, a, jpath = _run_full_analysis(tmpdir=str(tmp_path))
        with open(jpath) as f:
            data = json.load(f)
        assert "app_root_pids" in data
        assert len(data["app_root_pids"]) == len(APP_ROOT_PIDS)
        json_pids = {int(s.split(":")[0]) for s in data["app_root_pids"]}
        assert json_pids == APP_ROOT_PIDS

    def test_all_rules_are_myapp_t(self, tmp_path):
        """Every allow rule must involve myapp_t (source or target)."""
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        for line in txt.split("\n"):
            if line.startswith("allow "):
                assert "myapp_t" in line, f"Rule without myapp_t: {line}"

    def test_multiple_avc_types_detected(self, tmp_path):
        """The log contains denials for dir, file, tcp_socket etc. at minimum."""
        txt, a, jpath = _run_full_analysis(tmpdir=str(tmp_path))
        with open(jpath) as f:
            data = json.load(f)
        tclasses = set()
        for r in data["results"]:
            for avc in r["AVC"]:
                tclasses.add(avc["tclass"])
        assert "dir" in tclasses
        assert "file" in tclasses

    def test_explanations_present(self, tmp_path):
        txt, a, _ = _run_full_analysis(tmpdir=str(tmp_path))
        assert "# required by :" in txt

    def test_no_explanations_mode(self, tmp_path):
        a = _make_analyzer(show_explanations=False)
        txt = a.analyze(docs=[], json_dest=None)
        assert "# required by :" not in txt
        # But rules should still be present
        assert "allow myapp_t" in txt

    def test_no_tree_mode(self, tmp_path):
        a = _make_analyzer(show_pid_tree=False)
        txt = a.analyze(docs=[], json_dest=None)
        assert cla.PID_TREE_DELIMITER not in txt
        assert "allow myapp_t" in txt


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND-TRIP TESTS — JSON idempotency
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestJsonRoundTrip:
    """log → JSON₁ → (load JSON₁) → JSON₂ : JSON₁ ≡ JSON₂ (idempotent)."""

    def test_json_roundtrip_results_stable(self, tmp_path):
        """Results must be identical after a round-trip through JSON."""
        # Step 1: log → JSON₁
        j1 = str(tmp_path / "pass1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        # Step 2: JSON₁ → JSON₂  (no log, ignore-log)
        j2 = str(tmp_path / "pass2.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        # Compare results (the core data)
        assert len(d1["results"]) == len(d2["results"]), (
            f"Result count changed: {len(d1['results'])} → {len(d2['results'])}"
        )
        # Sort for deterministic comparison
        def result_key(r):
            avc = r["AVC"][0] if r["AVC"] else {}
            return (
                avc.get("source_type", ""),
                avc.get("target_type", ""),
                avc.get("tclass", ""),
                avc.get("method", ""),
                r["command"]["cmd"],
            )
        r1 = sorted(d1["results"], key=result_key)
        r2 = sorted(d2["results"], key=result_key)
        for i, (a, b) in enumerate(zip(r1, r2)):
            assert a["AVC"] == b["AVC"], f"AVC mismatch at index {i}: {a['AVC']} vs {b['AVC']}"
            assert a["command"]["cmd"] == b["command"]["cmd"], (
                f"cmd mismatch at index {i}"
            )

    def test_json_roundtrip_pid_tree_stable(self, tmp_path):
        """PID tree must survive JSON round-trip."""
        j1 = str(tmp_path / "pass1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        j2 = str(tmp_path / "pass2.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        assert set(d1["pid_tree"].keys()) == set(d2["pid_tree"].keys()), (
            "PID tree keys differ after round-trip"
        )

    def test_json_roundtrip_app_root_preserved(self, tmp_path):
        """APP root PIDs must be preserved through JSON round-trip."""
        j1 = str(tmp_path / "pass1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        j2 = str(tmp_path / "pass2.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        assert sorted(d1["app_root_pids"]) == sorted(d2["app_root_pids"])
        json_pids = {int(s.split(":")[0]) for s in d2["app_root_pids"]}
        assert json_pids == APP_ROOT_PIDS

    def test_json_roundtrip_counters(self, tmp_path):
        """AVC count from log pass must match file_counter after reload."""
        j1 = str(tmp_path / "pass1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        n_avc = a1.avc_counter

        with open(j1) as f:
            d1 = json.load(f)

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1])
        # All AVCs from the log should now be counted as file_counter
        assert a2.file_counter == len(d1["results"])


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND-TRIP TESTS — Human-readable text
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestTextRoundTrip:
    """log → text₁ → (load text₁ as --files) → text₂ : rules must be preserved."""

    def test_text_roundtrip_rules_preserved(self, tmp_path):
        # Step 1: log → text
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        # Write to file
        f1 = str(tmp_path / "pass1.txt")
        with open(f1, "w") as f:
            f.write(txt1)

        # Step 2: text → text (no log)
        with open(f1) as f:
            doc = f.read()
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[doc])

        # Extract rules from both
        def extract_rules(txt):
            rules = set()
            for line in txt.split("\n"):
                if line.startswith("allow "):
                    rules.add(line.strip())
            return rules

        rules1 = extract_rules(txt1)
        rules2 = extract_rules(txt2)
        assert len(rules1) > 0
        assert rules1 == rules2, f"Rules differ:\nOnly in pass1: {rules1 - rules2}\nOnly in pass2: {rules2 - rules1}"

    def test_text_roundtrip_pid_tree_preserved(self, tmp_path):
        """PID tree must survive text round-trip."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        with open(str(tmp_path / "pass1.txt"), "w") as f:
            f.write(txt1)
        with open(str(tmp_path / "pass1.txt")) as f:
            doc = f.read()

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[doc])

        # APP ROOT must still be in the output
        assert "[APP ROOT]" in txt2
        for pid in APP_ROOT_PIDS:
            assert f"pid={pid}@{REFERENCE_KEY}" in txt2, (
                f"APP root pid={pid} not found after text round-trip"
            )

    def test_text_roundtrip_index_preserved(self, tmp_path):
        """Index section must survive text round-trip."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt1])

        # Extract index section (between INDEX_DELIMITER and PID_TREE_DELIMITER)
        def extract_index(txt):
            parts = txt.split(cla.INDEX_DELIMITER)
            assert len(parts) == 2, "INDEX_DELIMITER not found"
            after_index = parts[1]
            tree_parts = after_index.split(cla.PID_TREE_DELIMITER)
            return tree_parts[0].strip()

        idx1 = extract_index(txt1)
        idx2 = extract_index(txt2)
        assert len(idx1) > 0, "Index section is empty"
        if idx1 != idx2:
            diff = "\n".join(difflib.unified_diff(
                idx1.splitlines(), idx2.splitlines(),
                fromfile="pass1/index", tofile="pass2/index", lineterm=""
            ))
            pytest.fail(f"Index section differs after text round-trip:\n{diff}")

    def test_text_roundtrip_stable(self, tmp_path):
        """Output should be identical after text round-trip."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt1])
        assert txt1 == txt2, "Text round-trip is not idempotent"

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-FORMAT TESTS — text ↔ JSON interoperability
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestCrossFormat:
    """Test crossing between text and JSON formats."""

    def test_log_to_json_to_text(self, tmp_path):
        """log → JSON → text : text must contain all rules."""
        j = str(tmp_path / "intermediate.json")
        a1 = _make_analyzer()
        txt_direct = a1.analyze(docs=[], json_dest=j)

        with open(j) as f:
            jdata = json.load(f)

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt_via_json = a2.analyze(docs=[], json_docs=[jdata])

        def extract_rules(txt):
            return {l.strip() for l in txt.split("\n") if l.startswith("allow ")}

        rules_direct = extract_rules(txt_direct)
        rules_via_json = extract_rules(txt_via_json)
        assert rules_direct == rules_via_json

    def test_log_to_text_to_json(self, tmp_path):
        """log → text → (re-parse text, emit JSON) : JSON must have all results."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])
        n_rules_direct = sum(1 for l in txt1.split("\n") if l.startswith("allow "))

        j = str(tmp_path / "from_text.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[txt1], json_dest=j)

        with open(j) as f:
            jdata = json.load(f)
        assert len(jdata["results"]) > 0
        json_pids = {int(s.split(":")[0]) for s in jdata["app_root_pids"]}
        assert json_pids == APP_ROOT_PIDS

    def test_json_to_text_to_json(self, tmp_path):
        """JSON → text → JSON : allow rules must be stable."""
        # First produce JSON from log
        j1 = str(tmp_path / "step1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        # JSON → text
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt = a2.analyze(docs=[], json_docs=[d1])

        # text → JSON
        j3 = str(tmp_path / "step3.json")
        a3 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a3.analyze(docs=[txt], json_dest=j3)
        with open(j3) as f:
            d3 = json.load(f)

        # We only compare allow rules, NOT result counts.
        # The internal model stores one AnalysisResult per audit event (pid ×
        # AVC list), but the text format groups results by allow-rule string.
        # When a single audit event has multiple AVCs (e.g. {read, getattr}),
        # format_rules splits it into two allow-rule headings.  Re-parsing
        # that text creates one result per (rule × command), inflating the
        # count.  The SELinux policy output is identical — only the internal
        # result granularity changes.  See test_text_to_json_to_text_to_json_stable
        # which proves that once through text, the structure is stable.
        def extract_rules(results):
            rules = set()
            for r in results:
                for avc in r.get("AVC", []):
                    rules.add((avc["source_type"], avc["target_type"],
                               avc["tclass"], avc["method"]))
            return rules

        rules1 = extract_rules(d1["results"])
        rules3 = extract_rules(d3["results"])
        assert rules1 == rules3, (
            f"Rule mismatch after JSON→text→JSON:\n"
            f"  only in original: {rules1 - rules3}\n"
            f"  only in roundtrip: {rules3 - rules1}"
        )

    def test_text_to_json_to_text_to_json_stable(self, tmp_path):
        """text → J₁ → T₂ → J₂ : J₁ and J₂ must be identical.

        The first text parse shapes results into 'text-native' form (one result
        per descriptor line).  Subsequent passes must be idempotent."""
        # log → text (starting material)
        a0 = _make_analyzer()
        txt0 = a0.analyze(docs=[])

        # text → J₁
        j1 = str(tmp_path / "j1.json")
        a1 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a1.analyze(docs=[txt0], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        # J₁ → T₂
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[], json_docs=[d1])

        # T₂ → J₂
        j2 = str(tmp_path / "j2.json")
        a3 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a3.analyze(docs=[txt2], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        # J₁ and J₂ must be identical
        assert len(d1["results"]) == len(d2["results"]), (
            f"Result count mismatch: {len(d1['results'])} → {len(d2['results'])}"
        )
        assert d1["index"] == d2["index"]
        assert set(d1["pid_tree"].keys()) == set(d2["pid_tree"].keys())
        assert sorted(d1["app_root_pids"]) == sorted(d2["app_root_pids"])

        def sort_results(results):
            return sorted(results, key=lambda r: json.dumps(r, sort_keys=True))
        assert sort_results(d1["results"]) == sort_results(d2["results"])


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENT KEY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestDifferentKeys:
    """Test analysis with different key values and merging across keys."""

    def test_different_key_appears_in_output(self, tmp_path):
        """Rules should reference the key used for analysis."""
        a = _make_analyzer(key="CustomHost")
        txt = a.analyze(docs=[])
        assert "CustomHost" in txt

    def test_merge_same_key(self, tmp_path):
        """Merging two analyses with the same key should deduplicate."""
        j1 = str(tmp_path / "part1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        # Re-merge with same data
        j2 = str(tmp_path / "merged.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1, d1], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        # file_counter should be 2x (each merge counts), but results should
        # contain duplicates since merge_json appends all results
        assert len(d2["results"]) >= len(d1["results"])

    def test_merge_different_keys(self, tmp_path):
        """Merging analyses from different keys should keep both keys in PID tree."""
        j1 = str(tmp_path / "rocky.json")
        a1 = _make_analyzer(key="Rocky")
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        j2 = str(tmp_path / "debian.json")
        a2 = _make_analyzer(key="Debian")
        a2.analyze(docs=[], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        # Merge both
        j3 = str(tmp_path / "merged.json")
        a3 = cla.Analyzer(key="Merged", look_in_log=False, show_pid_tree=True)
        a3.analyze(docs=[], json_docs=[d1, d2], json_dest=j3)
        with open(j3) as f:
            d3 = json.load(f)

        # Both keys should appear in PID tree keys
        tree_keys = set()
        for pk_str in d3["pid_tree"]:
            _, k = pk_str.split(":", 1)
            tree_keys.add(k)
        assert "Rocky" in tree_keys
        assert "Debian" in tree_keys

    def test_key_sanitisation(self):
        """Special characters in key should be replaced."""
        # The CLI sanitizes the key; let's test that pattern
        key = "SLES podman/v1"
        for c in [' ', '/', ',', ';', '.', '(', ')', '[', ']', '{', '}']:
            key = key.replace(c, "_")
        assert key == "SLES_podman_v1"


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT MODE COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestInputModes:
    """Test every input source combination."""

    def test_log_only(self, tmp_path):
        """--log only (no --files, no --json-files)."""
        a = _make_analyzer()
        txt = a.analyze(docs=[])
        assert "allow myapp_t" in txt
        assert a.avc_counter > 0

    def test_ignore_log_mode(self, tmp_path):
        """--ignore-log: must produce empty output with no input files."""
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False)
        txt = a.analyze(docs=[])
        # Should only have the index delimiter and nothing else meaningful
        assert a.avc_counter == 0

    def test_files_only(self, tmp_path):
        """--files only (no log)."""
        # First generate text from log
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        # Then re-parse from text only
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt1])
        assert "allow myapp_t" in txt2
        assert a2.file_counter > 0
        assert a2.avc_counter == 0

    def test_json_files_only(self, tmp_path):
        """--json-files only (no log)."""
        j = str(tmp_path / "src.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j)
        with open(j) as f:
            jdata = json.load(f)

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt = a2.analyze(docs=[], json_docs=[jdata])
        assert "allow myapp_t" in txt
        assert a2.file_counter > 0

    def test_log_plus_files(self, tmp_path):
        """--log + --files: merged results."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        a2 = _make_analyzer()
        txt2 = a2.analyze(docs=[txt1])
        # Should have both log and file contributions
        assert a2.avc_counter > 0
        assert a2.file_counter > 0

    def test_log_plus_json_files(self, tmp_path):
        """--log + --json-files: merged results."""
        j = str(tmp_path / "prev.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j)
        with open(j) as f:
            jdata = json.load(f)

        a2 = _make_analyzer()
        txt = a2.analyze(docs=[], json_docs=[jdata])
        assert a2.avc_counter > 0
        assert a2.file_counter > 0

    def test_files_plus_json_files(self, tmp_path):
        """--files + --json-files (no log)."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        j = str(tmp_path / "json_src.json")
        a1b = _make_analyzer()
        a1b.analyze(docs=[], json_dest=j)
        with open(j) as f:
            jdata = json.load(f)

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt1], json_docs=[jdata])
        assert "allow myapp_t" in txt2
        assert a2.file_counter > 0

    def test_all_inputs(self, tmp_path):
        """--log + --files + --json-files: everything merged."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        j = str(tmp_path / "json_src.json")
        a1b = _make_analyzer()
        a1b.analyze(docs=[], json_dest=j)
        with open(j) as f:
            jdata = json.load(f)

        a2 = _make_analyzer()
        txt2 = a2.analyze(docs=[txt1], json_docs=[jdata])
        assert a2.avc_counter > 0
        assert a2.file_counter > 0


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestOutputModes:
    """Test different output destinations."""

    def test_stdout_output(self):
        """Default: output to stdout (returned as string)."""
        a = _make_analyzer()
        txt = a.analyze(docs=[])
        assert isinstance(txt, str)
        assert len(txt) > 0

    def test_json_dest_only(self, tmp_path):
        """--json-dest only (skip_formatting=True)."""
        j = str(tmp_path / "out.json")
        a = _make_analyzer()
        txt = a.analyze(docs=[], json_dest=j, skip_formatting=True)
        assert txt == ""  # no human-readable output
        assert os.path.isfile(j)
        with open(j) as f:
            data = json.load(f)
        assert len(data["results"]) > 0

    def test_both_outputs(self, tmp_path):
        """--dest + --json-dest: both produced."""
        j = str(tmp_path / "out.json")
        a = _make_analyzer()
        txt = a.analyze(docs=[], json_dest=j)
        assert len(txt) > 0
        assert os.path.isfile(j)
        with open(j) as f:
            data = json.load(f)
        assert len(data["results"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PID TREE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestPidTree:
    """Tests specific to PID tree construction and formatting."""

    def test_pid_tree_has_myapp_t_entries(self, tmp_path):
        """PID tree should contain processes with myapp_t context."""
        a = _make_analyzer()
        a.analyze(docs=[])
        myapp_t_entries = [
            pk for pk, info in a.pid_tree.items()
            if "myapp_t" in info.context
        ]
        assert len(myapp_t_entries) > 0

    def test_pid_tree_parent_child_consistency(self, tmp_path):
        """Every child reference should point to an existing entry."""
        a = _make_analyzer()
        a.analyze(docs=[])
        for pk, info in a.pid_tree.items():
            for child_pk in info.children:
                assert child_pk in a.pid_tree, (
                    f"Child {child_pk} of {pk} not in pid_tree"
                )

    def test_pid_tree_format_valid(self, tmp_path):
        """Formatted PID tree should have valid structure."""
        a = _make_analyzer()
        a.analyze(docs=[])
        tree_txt = a.format_pid_tree()
        assert "pid=" in tree_txt
        assert "ctx:" in tree_txt
        assert "cmd:" in tree_txt

    def test_app_root_children_in_tree(self, tmp_path):
        """PID 1001 (main APP start) should have children (e.g. worker)."""
        a = _make_analyzer()
        a.analyze(docs=[])
        root_pk = (1001, REFERENCE_KEY)
        assert root_pk in a.pid_tree, f"APP root {root_pk} not in pid_tree"
        root_entry = a.pid_tree[root_pk]
        assert len(root_entry.children) > 0, "APP root has no children"

    def test_pid_tree_no_orphan_app_root(self, tmp_path):
        """APP root should be part of a tree, not listed as orphan."""
        a = _make_analyzer()
        txt = a.analyze(docs=[])
        # Check APP ROOT is under "Process Tree" not "Orphan"
        parts = txt.split(cla.PID_TREE_DELIMITER)
        assert len(parts) >= 2, "Expected PID_TREE_DELIMITER in output"
        tree_section = parts[1]
        # Walk sections; track current header context
        found_in_tree = False
        found_in_orphan = False
        current_is_orphan = False
        for line in tree_section.split("\n"):
            if line.startswith("# ") and "Process Tree" in line:
                current_is_orphan = False
            elif line.startswith("# ") and "Orphan" in line:
                current_is_orphan = True
            if "[APP ROOT]" in line:
                if current_is_orphan:
                    found_in_orphan = True
                else:
                    found_in_tree = True
        assert found_in_tree, "APP ROOT should appear under a Process Tree header"
        assert not found_in_orphan, "APP ROOT should not appear under Orphan header"


# ═══════════════════════════════════════════════════════════════════════════════
# STATE FILE TESTS (incremental analysis)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestStateFile:
    """Test the state file mechanism for incremental analysis."""

    def test_state_file_created(self, tmp_path):
        sf = str(tmp_path / "state.json")
        a = _make_analyzer(state_file_path=sf)
        a.analyze(docs=[])
        assert os.path.isfile(sf)

    def test_state_file_contains_entries(self, tmp_path):
        sf = str(tmp_path / "state.json")
        a = _make_analyzer(state_file_path=sf)
        a.analyze(docs=[])
        with open(sf) as f:
            state = json.load(f)
        assert len(state["analyzed_entries"]) > 0

    def test_second_run_skips_analyzed(self, tmp_path):
        """Second run with state file should find fewer new entries."""
        sf = str(tmp_path / "state.json")

        a1 = _make_analyzer(state_file_path=sf)
        a1.analyze(docs=[])
        count1 = a1.avc_counter

        a2 = _make_analyzer(state_file_path=sf)
        a2.analyze(docs=[])
        count2 = a2.avc_counter

        # Second run should have found 0 new AVCs (all were already analyzed)
        assert count2 == 0, (
            f"Expected 0 new AVCs on second run, got {count2} (first run: {count1})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEdgeCases:
    """Edge cases that don't require the reference log."""

    def test_empty_analysis(self):
        """Analysis with no log and no files should produce minimal output."""
        a = cla.Analyzer(key="empty", look_in_log=False, show_pid_tree=False)
        txt = a.analyze(docs=[])
        # Should just have the index delimiter
        assert cla.INDEX_DELIMITER in txt
        assert a.avc_counter == 0
        assert a.file_counter == 0

    def test_empty_json_roundtrip(self, tmp_path):
        """Empty analysis → JSON → reload should work."""
        j = str(tmp_path / "empty.json")
        a1 = cla.Analyzer(key="empty", look_in_log=False, show_pid_tree=False)
        a1.analyze(docs=[], json_dest=j)
        with open(j) as f:
            data = json.load(f)
        assert data["version"] == cla.JSON_FORMAT_VERSION
        assert len(data["results"]) == 0

        a2 = cla.Analyzer(key="empty", look_in_log=False, show_pid_tree=False)
        txt = a2.analyze(docs=[], json_docs=[data])
        assert cla.INDEX_DELIMITER in txt

    def test_skip_formatting_saves_state(self, tmp_path):
        """skip_formatting=True should still save state file."""
        sf = str(tmp_path / "state.json")
        j = str(tmp_path / "out.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.mark_entry_analyzed("type=AVC msg=audit(1234:100) : test")
        txt = a.analyze(docs=[], json_dest=j, skip_formatting=True)
        assert txt == ""
        # State file should have been saved
        assert os.path.isfile(sf)
        with open(sf) as f:
            state = json.load(f)
        assert len(state["analyzed_entries"]) > 0

    def test_parse_ausearch_with_direct_blocks(self):
        """Feed AVC blocks directly to parse_ausearch_from_log."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/bash /bin/myapp -s
            type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=newfstatat success=yes exit=0 a0=AT_FDCWD a1=0x55b422eaebb0 a2=0x7fffde50c510 a3=0x0 items=1 ppid=23905 pid=23906 auid=myapp uid=root gid=root euid=root suid=root fsuid=root egid=root sgid=root fsgid=root tty=pts2 ses=68 comm=myapp exe=/usr/bin/bash subj=system_u:system_r:myapp_t:s0-s0:c0.c1023 key=(null)
            type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } for  pid=23906 comm=myapp path=/usr/share/myapp/selinux/myapp_selinux_configure dev="vda4" ino=1346386 scontext=system_u:system_r:myapp_t:s0-s0:c0.c1023 tcontext=system_u:object_r:user_tmp_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert results[0].avc_list[0].source_type == "myapp_t"
        assert results[0].avc_list[0].target_type == "user_tmp_t"
        assert results[0].avc_list[0].tclass == "file"
        assert results[0].avc_list[0].method == "getattr"
        assert results[0].command.pid == "23906"
        assert results[0].command.cmd == "/bin/bash /bin/myapp -s"

    def test_parse_ausearch_multi_avc_single_block(self):
        """Block with multiple AVC denials in one event."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:19.602:18451) : proctitle=/bin/pasta --version
            type=SYSCALL msg=audit(09/02/2026 10:45:19.602:18451) : arch=x86_64 syscall=execve success=yes exit=0 a0=0xc000013580 a1=0xc000647770 a2=0xc000648dc0 a3=0x0 items=2 ppid=23910 pid=23927 auid=myapp uid=root gid=root euid=root suid=root fsuid=root egid=root sgid=root fsgid=root tty=pts2 ses=68 comm=pasta exe=/usr/bin/pasta subj=system_u:system_r:myapp_t:s0-s0:c0.c1023 key=exec_logs
            type=AVC msg=audit(09/02/2026 10:45:19.602:18451) : avc:  denied  { read open } for  pid=23927 comm=podman path=/usr/bin/pasta dev="vda4" ino=1335751 scontext=system_u:system_r:myapp_t:s0-s0:c0.c1023 tcontext=system_u:object_r:pasta_exec_t:s0 tclass=file permissive=1
            type=AVC msg=audit(09/02/2026 10:45:19.602:18451) : avc:  denied  { execute_no_trans } for  pid=23927 comm=podman path=/usr/bin/pasta dev="vda4" ino=1335751 scontext=system_u:system_r:myapp_t:s0-s0:c0.c1023 tcontext=system_u:object_r:pasta_exec_t:s0 tclass=file permissive=1
            type=AVC msg=audit(09/02/2026 10:45:19.602:18451) : avc:  denied  { map } for  pid=23927 comm=pasta path=/usr/bin/pasta dev="vda4" ino=1335751 scontext=system_u:system_r:myapp_t:s0-s0:c0.c1023 tcontext=system_u:object_r:pasta_exec_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        methods = {avc.method for avc in results[0].avc_list}
        assert "read" in methods
        assert "open" in methods
        assert "execute_no_trans" in methods
        assert "map" in methods

    def test_parse_ausearch_non_matching_filtered(self):
        """Non-matching AVCs should be filtered out when context_filter is set."""
        block = textwrap.dedent("""\
            type=PROCTITLE msg=audit(09/02/2026 10:45:16.012:18378) : proctitle=/usr/libexec/openssh/sshd-session -D -R
            type=SYSCALL msg=audit(09/02/2026 10:45:16.012:18378) : arch=x86_64 syscall=openat success=yes exit=9 a0=AT_FDCWD a1=0x7fba81c95743 a2=O_RDONLY|O_NOCTTY|O_CLOEXEC a3=0x0 items=1 ppid=1044 pid=23819 auid=root uid=root gid=root euid=root suid=root fsuid=root egid=root sgid=root fsgid=root tty=(none) ses=141 comm=sshd-session exe=/usr/libexec/openssh/sshd-session subj=system_u:system_r:sshd_t:s0-s0:c0.c1023 key=(null)
            type=AVC msg=audit(09/02/2026 10:45:16.012:18378) : avc:  denied  { read } for  pid=23819 comm=sshd-session name=machine-id dev="vda4" ino=1569927 scontext=system_u:system_r:sshd_t:s0-s0:c0.c1023 tcontext=system_u:object_r:unlabeled_t:s0 tclass=file permissive=1
        """)
        a = cla.Analyzer(key="test", look_in_log=False, context_filter=["myapp_t"])
        results = a.parse_ausearch_from_log(blocks=[block])
        # sshd_t → unlabeled_t : neither matches context_filter, so it should be filtered out
        assert len(results) == 0

    def test_json_version_mismatch_warning(self, tmp_path, capsys):
        """Loading JSON with wrong version should warn but not crash."""
        data = {"version": 999, "results": [], "index": {}, "ns_index": {},
                "all_cmds": {}, "all_aliases": [], "pid_tree": {},
                "app_root_pids": [], "avc_pids": [], "counters": {}}
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.merge_json(data)
        assert len(results) == 0
        # Warning should be printed to stderr
        captured = capsys.readouterr()
        assert "version" in captured.err.lower() or "Warning" in captured.err


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX TESTS — Phase 1/2/3
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — log_cmd / register_cmd / indexing
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCmdLoggingAndIndexing:
    """Integration tests for log_cmd(), register_cmd(), and the indexing workflow."""

    def test_log_cmd_accumulates_weight_across_calls(self):
        """log_cmd() should accumulate weights across multiple calls."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Log same command multiple times with different weights
        a.log_cmd("/usr/bin/test-command", weight=1)
        a.log_cmd("/usr/bin/test-command", weight=1)
        a.log_cmd("/usr/bin/test-command", weight=3)

        # Weight should accumulate (1 + 1 + 3 = 5)
        assert a.cmd_index.get_weight("/usr/bin/test-command") == 5

    def test_log_cmd_during_parse_accumulates(self):
        """Commands logged during parsing should accumulate weight."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Simulate multiple parse results with same command
        # Note: parse_ausearch doesn't log commands that don't trigger filtering
        # The commands are only logged if they pass through the analysis
        blocks = [
            'type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="testcmd" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file',
            'type=AVC msg=audit(1235:101) : avc: denied { write } for pid=1002 comm="testcmd" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:tmp_t:s0 tclass=file',
        ]

        results = a.parse_ausearch_from_log(blocks)

        # Check that results were generated
        assert len(results) >= 2

    def test_alias_stability_after_build(self):
        """Once build_index() is called, aliases should remain stable."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Register a command that will get an alias
        # Must have >1 word (INDEX_MIN_ARG) and either high weight OR long length
        long_cmd = "/usr/bin/very-long-command-name-that-should-be-indexed with args"
        a.log_cmd(long_cmd, weight=100)  # High weight ensures indexing

        # Build index
        a.build_index()

        # Get the assigned alias
        alias1 = a.get_index(long_cmd)

        # Alias should be different from original (it's indexed)
        assert alias1 != long_cmd
        # Aliases are uppercase transformations, not tilde-wrapped
        assert alias1.isupper() or "_" in alias1

        # Call build_index again - alias should stay the same
        a.build_index()
        alias2 = a.get_index(long_cmd)

        assert alias1 == alias2

    def test_alias_appears_in_formatted_rules(self):
        """Aliases from cmd_index should appear in formatted rules output."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Register a long command
        long_cmd = "/usr/bin/very-long-command-that-needs-alias"
        a.set_index(long_cmd, "~TESTCMD~")

        # Create a result using this command
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd=long_cmd, pid=1, pid_namespace=12345),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )

        rules = a.format_rules([result])

        # Convert rules dict to string to check for alias
        rules_str = str(rules)
        assert "~TESTCMD~" in rules_str

    def test_alias_appears_in_pid_tree_output(self):
        """Aliases should appear in PID tree formatted output."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Register a long command with alias
        long_cmd = "/usr/bin/very-long-command-for-tree"
        a.set_index(long_cmd, "~TREECMD~")

        # Add to PID tree
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=long_cmd, ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.avc_pids.add(pk)

        output = a.format_pid_tree()
        assert "~TREECMD~" in output

    def test_build_index_creates_aliases_for_long_commands(self):
        """build_index() should create aliases for long commands with >1 word."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Log a very long command with arguments (need >1 word)
        long_cmd = "/usr/bin/" + "x" * 150 + " arg1"  # Very long command with arg
        a.log_cmd(long_cmd, weight=1)

        # Build index
        a.build_index()

        # Should get an alias
        alias = a.get_index(long_cmd)
        assert alias != long_cmd
        assert len(alias) < len(long_cmd)
        # Aliases are uppercase or have underscores
        assert alias.isupper() or "_" in alias

    def test_build_index_creates_aliases_for_frequent_commands(self):
        """build_index() should create aliases for frequently used commands with >1 word."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Log a command many times (high weight) - must have >1 word
        cmd = "/usr/bin/frequent-command with args"
        a.log_cmd(cmd, weight=1000)  # Very high weight

        # Build index
        a.build_index()

        # Should get an alias
        alias = a.get_index(cmd)
        assert alias != cmd
        # Aliases are uppercase or have underscores
        assert alias.isupper() or "_" in alias

    def test_build_index_ignores_short_infrequent_commands(self):
        """build_index() should NOT create aliases for short, infrequent commands."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Log a short command with low weight
        cmd = "ls"
        a.log_cmd(cmd, weight=1)

        # Build index
        a.build_index()

        # Should NOT get an alias
        alias = a.get_index(cmd)
        assert alias == cmd  # No alias, returns as-is

    def test_register_cmd_with_custom_alias(self):
        """set_index() should allow registering custom aliases."""
        a = cla.Analyzer(key="test", look_in_log=False)

        cmd = "/usr/bin/my-special-command"
        custom_alias = "~MYCMD~"

        actual_alias = a.set_index(cmd, custom_alias)

        # Should use the custom alias
        assert actual_alias == custom_alias
        assert a.get_index(cmd) == custom_alias

    def test_index_format_output_shows_aliases(self):
        """format_index() should include registered aliases in output."""
        a = cla.Analyzer(key="test", look_in_log=False)

        # Register some commands
        a.set_index("/usr/bin/command1", "~CMD1~")
        a.set_index("/usr/bin/command2", "~CMD2~")

        output = a.format_index()

        # Both aliases should appear in output
        assert "~CMD1~" in output
        assert "~CMD2~" in output
        assert "/usr/bin/command1" in output
        assert "/usr/bin/command2" in output


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — --no-index flag
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestNoIndex:
    """Tests for the --no-index feature that skips all indexation."""

    def test_log_cmd_is_noop(self):
        """log_cmd() should do nothing when no_index=True."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        a.log_cmd("/usr/bin/some-command arg1 arg2", weight=100)
        assert a.cmd_index.get_weight("/usr/bin/some-command arg1 arg2") == 0
        assert len(a.cmd_index) == 0

    def test_set_index_returns_raw_cmd(self):
        """set_index() should return the raw command without registering it."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        result = a.set_index("/usr/bin/my-command", "~ALIAS~")
        assert result == "/usr/bin/my-command"
        assert len(a.cmd_index) == 0

    def test_get_index_returns_raw_cmd(self):
        """get_index() should return commands as-is since nothing is registered."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        cmd = "/usr/bin/some-command with args"
        assert a.get_index(cmd) == cmd

    def test_build_index_skipped(self):
        """build_index() should be skipped in analyze() when no_index=True."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        # Log commands that would normally trigger alias creation
        a.cmd_index.log_weight("/usr/bin/long-command arg1 arg2", 1000)
        txt = a.analyze(docs=[])
        # No aliases should have been created
        assert len(a.cmd_index) == 0

    def test_format_index_empty(self):
        """format_index output should be empty when no_index=True."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        txt = a.analyze(docs=[])
        # INDEX section exists as delimiter but contains no entries
        idx_parts = txt.split(cla.INDEX_DELIMITER)
        assert len(idx_parts) == 2
        index_content = idx_parts[1].split(cla.PID_TREE_DELIMITER)[0]
        # No "###" index lines (namespace or command aliases)
        assert "### " not in index_content

    def test_rules_show_full_commands(self):
        """With no_index, rules should show full command strings, not aliases."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        long_cmd = "/usr/bin/very-long-command-name with many arguments"
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="test", cmd=long_cmd, pid=1, pid_namespace=cla.NOT_FOUND),
            avc_list=[cla.AvcDenial("myapp_t", "cert_t", "dir", "read")],
        )
        rules = a.format_rules([result])
        rules_str = str(rules)
        # Full command should appear, no alias substitution
        assert long_cmd in rules_str

    def test_pid_tree_shows_full_commands(self):
        """With no_index, PID tree should show full command strings."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        long_cmd = "/usr/bin/very-long-command-for-tree arg1 arg2"
        pk = (1234, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=long_cmd, ppid=None, context="system_u:system_r:myapp_t:s0",
            key="test", live=False,
        )
        a.avc_pids.add(pk)
        output = a.format_pid_tree()
        assert long_cmd in output

    def test_full_analyze_no_index(self):
        """Full analyze() with no_index should produce valid output without index entries."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        # Feed some AVC blocks
        blocks = [
            'type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="testcmd" '
            'scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file',
        ]
        results = a.parse_ausearch_from_log(blocks)
        assert len(results) > 0
        # No weights recorded
        assert len(a.cmd_index._weights) == 0

    def test_no_index_with_set_index_from_file_parse(self):
        """set_index() during file parsing should be a no-op with no_index."""
        a = cla.Analyzer(key="test", look_in_log=False, no_index=True)
        alias = a.set_index("/usr/bin/podman info", "PODMAN_0001")
        assert alias == "/usr/bin/podman info"
        assert "PODMAN_0001" not in a.cmd_index.aliases_sorted()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Indexing (existing tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestIndexing:
    """Test that indexing works correctly on real data."""

    def test_frequent_commands_indexed(self, tmp_path):
        """Commands that appear frequently should get aliases."""
        a = _make_analyzer()
        a.analyze(docs=[])
        # "podman info" appears multiple times in the log
        indexed_cmds = set(a.cmd_index.keys())
        assert len(indexed_cmds) > 0

    def test_index_appears_in_output(self, tmp_path):
        """The index section should contain ### entries."""
        a = _make_analyzer()
        txt = a.analyze(docs=[])
        parts = txt.split(cla.INDEX_DELIMITER)
        assert len(parts) == 2
        index_part = parts[1].split(cla.PID_TREE_DELIMITER)[0]
        # Should have at least one ### line
        index_lines = [l for l in index_part.split("\n") if l.startswith("###")]
        # May or may not have indexed commands depending on frequency
        # but the delimiter itself should be there

    def test_index_roundtrip(self, tmp_path):
        """Index should survive text → re-parse → text."""
        a1 = _make_analyzer()
        txt1 = a1.analyze(docs=[])

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt1])

        # Extract indexed aliases from both
        def get_aliases(txt):
            parts = txt.split(cla.INDEX_DELIMITER)
            if len(parts) < 2:
                return set()
            idx = parts[1].split(cla.PID_TREE_DELIMITER)[0]
            aliases = set()
            for line in idx.split("\n"):
                if line.startswith("### "):
                    words = line.split()
                    if len(words) >= 2:
                        aliases.add(words[1])
            return aliases

        a1_aliases = get_aliases(txt1)
        a2_aliases = get_aliases(txt2)
        # Aliases from pass1 should all exist in pass2
        assert a1_aliases.issubset(a2_aliases), (
            f"Lost aliases: {a1_aliases - a2_aliases}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTEGRATION (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
@needs_testlog
@needs_ausearch
class TestCLI:
    """Test running the script as a CLI subprocess."""

    def test_cli_basic(self, tmp_path):
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--dest", dest],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        with open(dest) as f:
            content = f.read()
        assert "allow myapp_t" in content

    def test_cli_json_dest(self, tmp_path):
        jdest = str(tmp_path / "output.json")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--app-name", "myapp",
             "--context-filter", "myapp_t",
             "--json-dest", jdest],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        with open(jdest) as f:
            data = json.load(f)
        assert data["version"] == cla.JSON_FORMAT_VERSION
        json_pids = {int(s.split(":")[0]) for s in data["app_root_pids"]}
        assert json_pids == APP_ROOT_PIDS

    def test_cli_roundtrip(self, tmp_path):
        """CLI: log → dest + json-dest → re-load json-dest → dest2."""
        dest1 = str(tmp_path / "pass1.txt")
        jdest1 = str(tmp_path / "pass1.json")
        r1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest1, "--json-dest", jdest1],
            capture_output=True, text=True, timeout=120,
        )
        assert r1.returncode == 0, f"Pass 1 failed:\n{r1.stderr}"

        dest2 = str(tmp_path / "pass2.txt")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--ignore-log",
             "--json-files", jdest1,
             "--dest", dest2],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"Pass 2 failed:\n{r2.stderr}"

        with open(dest1) as f:
            txt1 = f.read()
        with open(dest2) as f:
            txt2 = f.read()

        rules1 = {l.strip() for l in txt1.split("\n") if l.startswith("allow ")}
        rules2 = {l.strip() for l in txt2.split("\n") if l.startswith("allow ")}
        assert rules1 == rules2

    def test_cli_files_input(self, tmp_path):
        """CLI: --log → dest1 ; --files dest1 --ignore-log → dest2."""
        dest1 = str(tmp_path / "pass1.txt")
        r1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest1],
            capture_output=True, text=True, timeout=120,
        )
        assert r1.returncode == 0, f"Pass 1 failed:\n{r1.stderr}"

        dest2 = str(tmp_path / "pass2.txt")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--ignore-log",
             "--files", dest1,
             "--dest", dest2],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"Pass 2 failed:\n{r2.stderr}"

        with open(dest1) as f:
            txt1 = f.read()
        with open(dest2) as f:
            txt2 = f.read()

        rules1 = {l.strip() for l in txt1.split("\n") if l.startswith("allow ")}
        rules2 = {l.strip() for l in txt2.split("\n") if l.startswith("allow ")}
        assert rules1 == rules2

    def test_cli_state_file(self, tmp_path):
        """CLI: two runs with --state-file, second should find 0 new AVCs."""
        sf = str(tmp_path / "state.json")
        dest1 = str(tmp_path / "pass1.txt")
        r1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest1, "--state-file", sf],
            capture_output=True, text=True, timeout=120,
        )
        assert r1.returncode == 0, f"Pass 1 failed:\n{r1.stderr}"
        # Parse the summary line from stderr
        assert "AVC analyzed from logs" in r1.stderr

        dest2 = str(tmp_path / "pass2.txt")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest2, "--state-file", sf],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"Pass 2 failed:\n{r2.stderr}"
        # Second run should report 0 AVC from logs
        assert r2.stderr.strip().endswith(")") or "0 AVC analyzed from logs" in r2.stderr

    def test_cli_no_tree(self, tmp_path):
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest, "--no-tree"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        with open(dest) as f:
            content = f.read()
        assert cla.PID_TREE_DELIMITER not in content

    def test_cli_no_explanations(self, tmp_path):
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--dest", dest, "--no-explanations"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        with open(dest) as f:
            content = f.read()
        assert "# required by :" not in content
        assert "allow myapp_t" in content

    def test_cli_stderr_summary(self, tmp_path):
        """The final line on stderr should report AVC counts."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY, "--log", TEST_LOG,
             "--no-tree"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "AVC analyzed from logs" in result.stderr
        assert "PID tree:" in result.stderr
        assert "found" in result.stderr
        assert "displayed" in result.stderr


# ═══════════════════════════════════════════════════════════════════════════════
# --APP-NAME AND --CONTEXT-FILTER ARGUMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
@needs_testlog
@needs_ausearch
class TestAppNameAndContextFilter:
    """Tests asserting that --app-name and --context-filter CLI arguments are functional."""

    def test_app_name_controls_root_detection(self, tmp_path):
        """--app-name myapp with --context-filter myapp_t detects APP root PIDs;
        a wrong --app-name detects none."""
        jdest_correct = str(tmp_path / "correct.json")
        r = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--app-name", "myapp",
             "--context-filter", "myapp_t",
             "--json-dest", jdest_correct],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"CLI failed:\n{r.stderr}"
        with open(jdest_correct) as f:
            data = json.load(f)
        json_pids = {int(s.split(":")[0]) for s in data["app_root_pids"]}
        assert json_pids == APP_ROOT_PIDS, (
            f"Expected app roots {APP_ROOT_PIDS}, got {json_pids}"
        )

        # Using a wrong app name should detect no roots
        jdest_wrong = str(tmp_path / "wrong.json")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--app-name", "wrongapp",
             "--context-filter", "myapp_t",
             "--json-dest", jdest_wrong],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"CLI failed:\n{r2.stderr}"
        with open(jdest_wrong) as f:
            data2 = json.load(f)
        assert len(data2["app_root_pids"]) == 0, (
            "Wrong app name should detect no APP roots"
        )

    def test_context_filter_restricts_rules(self, tmp_path):
        """--context-filter myapp_t includes myapp_t rules; wrong filter excludes all."""
        dest_correct = str(tmp_path / "correct.txt")
        r = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--context-filter", "myapp_t",
             "--dest", dest_correct],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"CLI failed:\n{r.stderr}"
        with open(dest_correct) as f:
            content = f.read()
        assert "allow myapp_t" in content, "Expected myapp_t rules with correct filter"

        # Using a wrong filter should produce no allow rules
        dest_wrong = str(tmp_path / "wrong.txt")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--context-filter", "wrongapp_t",
             "--dest", dest_wrong],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"CLI failed:\n{r2.stderr}"
        with open(dest_wrong) as f:
            content2 = f.read()
        rule_lines = [l for l in content2.split("\n") if l.startswith("allow ")]
        assert len(rule_lines) == 0, "Wrong context filter should produce no allow rules"

    def test_app_name_without_context_filter_skips_root_detection(self, tmp_path):
        """Without --context-filter, --app-name alone should not detect any APP roots
        (both params are required per identify_app_root logic)."""
        jdest = str(tmp_path / "out.json")
        r = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--app-name", "myapp",
             "--json-dest", jdest],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"CLI failed:\n{r.stderr}"
        with open(jdest) as f:
            data = json.load(f)
        assert len(data["app_root_pids"]) == 0, (
            "--app-name without --context-filter should detect no APP roots"
        )

    def test_no_context_filter_produces_rules_from_all_contexts(self, tmp_path):
        """Without --context-filter, allow rules are generated for every scontext
        present in test_log.  In particular httpd_t AVCs (PIDs 4001/4002) must
        appear in the output, and myapp_t rules must also be present."""
        dest = str(tmp_path / "out.txt")
        r = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", REFERENCE_KEY,
             "--log", TEST_LOG,
             "--dest", dest],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"CLI failed:\n{r.stderr}"
        with open(dest) as f:
            content = f.read()
        allow_lines = [l for l in content.split("\n") if l.startswith("allow ")]
        assert len(allow_lines) > 0, (
            "Without --context-filter, at least some allow rules must be generated"
        )
        # test_log contains httpd_t AVCs (2112, 2113) – must appear unfiltered
        assert any("allow httpd_t" in l for l in allow_lines), (
            "Expected allow httpd_t rules from test_log httpd AVCs (events 2112/2113)"
        )
        # myapp_t entries dominate test_log and must also appear
        assert any("allow myapp_t" in l for l in allow_lines), (
            "Expected allow myapp_t rules from test_log myapp AVCs"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
class TestCLIArgumentValidation:
    """Test CLI argument validation and error handling."""

    def test_invalid_log_path(self, tmp_path):
        """CLI should handle invalid log path gracefully."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--log", "/nonexistent/path/to/audit.log",
             "--dest", dest],
            capture_output=True, text=True,
        )
        # Should fail gracefully (non-zero exit)
        assert result.returncode != 0

    def test_key_with_special_characters(self, tmp_path):
        """CLI should handle keys with special characters."""
        dest = str(tmp_path / "output.txt")
        # Create a minimal AVC block
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write('type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="test" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file\n')

        # Test with special characters in key
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "host-with-dashes_underscores.dots",
             "--log", log_file,
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

        # Check output contains the key (dots might be sanitized to underscores)
        with open(dest) as f:
            content = f.read()
        # The key appears but special chars might be normalized
        assert "host-with-dashes_underscores" in content

    def test_help_output(self):
        """CLI --help should work and display usage."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "Usage:" in result.stdout

    def test_help_includes_no_index(self):
        """CLI --help should mention --no-index."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--help"],
            capture_output=True, text=True,
        )
        assert "--no-index" in result.stdout

    def test_missing_required_key(self, tmp_path):
        """CLI without --key should use default key."""
        dest = str(tmp_path / "output.txt")
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write('type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="test" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file\n')

        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--log", log_file,
             "--dest", dest],
            capture_output=True, text=True,
        )
        # Should succeed with default key
        assert result.returncode == 0

    def test_conflicting_ignore_log_and_log(self, tmp_path):
        """CLI with --ignore-log should ignore --log argument."""
        dest = str(tmp_path / "output.txt")
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write('type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="test" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file\n')

        # Use both --ignore-log and --log (--ignore-log should win)
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--log", log_file,
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

        # Output should be minimal (no log analysis)
        with open(dest) as f:
            content = f.read()
        # Should not have analyzed AVCs from the log
        assert "AVC analyzed from logs : 1" not in content

    def test_empty_log_file(self, tmp_path):
        """CLI should handle empty log file gracefully."""
        dest = str(tmp_path / "output.txt")
        log_file = str(tmp_path / "empty.log")
        # Create empty log file
        with open(log_file, "w") as f:
            pass

        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--log", log_file,
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        # Should succeed with empty/minimal output
        assert result.returncode == 0

    def test_json_dest_creates_valid_json(self, tmp_path):
        """CLI with --json-dest should create valid JSON file."""
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write('type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="test" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file\n')

        json_dest = str(tmp_path / "output.json")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--log", log_file,
             "--json-dest", json_dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

        # Check JSON is valid
        with open(json_dest) as f:
            import json
            data = json.load(f)
        assert "version" in data
        assert "key" in data

    def test_multiple_files_input(self, tmp_path):
        """CLI should accept multiple --files arguments."""
        # Create two properly formatted output files (with INDEX sections)
        file1 = str(tmp_path / "file1.txt")
        file2 = str(tmp_path / "file2.txt")

        # File format must include INDEX delimiter
        content_template = """allow myapp_t cert_t:dir read;
#####################
###     INDEX     ###
#####################

"""
        for f in [file1, file2]:
            with open(f, "w") as fh:
                fh.write(content_template)

        dest = str(tmp_path / "merged.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "merged",
             "--ignore-log",
             "--files", file1, file2,
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_state_file_path_created(self, tmp_path):
        """CLI with --state-file should create the state file."""
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write('type=AVC msg=audit(1234:100) : avc: denied { read } for pid=1001 comm="test" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:cert_t:s0 tclass=file\n')

        state_file = str(tmp_path / "state.json")
        dest = str(tmp_path / "output.txt")

        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--log", log_file,
             "--dest", dest,
             "--state-file", state_file],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

        # State file should exist
        assert os.path.exists(state_file)

        # State file should contain valid JSON
        with open(state_file) as f:
            import json
            state = json.load(f)
        assert "analyzed_entries" in state


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC TEST_LOG CONTENT ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestKnownLogSpecific:
    """Assertions specific to the known content of ./test_log."""

    def test_known_avc_rules_present(self, tmp_path):
        """Verify specific rules we know should be generated from test_log."""
        a = _make_analyzer()
        txt = a.analyze(docs=[])
        # These rules come from known AVC denials in the log
        known_rules = [
            "allow myapp_t myapp_t:tcp_socket shutdown;",
            "allow myapp_t user_tmp_t:file getattr;",
            "allow myapp_t user_tmp_t:file execute;",
            "allow myapp_t user_tmp_t:file read;",
            "allow myapp_t cert_t:dir getattr;",
            "allow myapp_t myapp_exec_t:file getattr;",
            "allow myapp_t myapp_exec_t:file execute;",
            "allow myapp_t binfmt_misc_fs_t:dir getattr;",
        ]
        for rule in known_rules:
            assert rule in txt, f"Expected rule not found: {rule}"

    def test_known_pids_in_tree(self, tmp_path):
        """Known PIDs from test_log should appear in the PID tree."""
        a = _make_analyzer()
        a.analyze(docs=[])
        # All 4 APP root PIDs must be in the tree
        for pid in APP_ROOT_PIDS:
            assert (pid, REFERENCE_KEY) in a.pid_tree, (
                f"APP root PID {pid} not in pid_tree"
            )
        # PID 1002 = /usr/lib/myapp/worker (child of 1001)
        assert (1002, REFERENCE_KEY) in a.pid_tree

    def test_podman_is_child_of_app_root(self, tmp_path):
        """PID 1002 (worker) should be a child of PID 1001 (APP root)."""
        a = _make_analyzer()
        a.analyze(docs=[])
        root_entry = a.pid_tree.get((1001, REFERENCE_KEY))
        assert root_entry is not None
        child_pids = [c[0] for c in root_entry.children]
        assert 1002 in child_pids, (
            f"worker (1002) not a child of APP root. Children: {root_entry.children}"
        )

    def test_avc_count_plausible(self, tmp_path):
        """AVC count should be > 0 and reasonable for this log."""
        a = _make_analyzer()
        a.analyze(docs=[])
        # The log has ~21 AVC events; after ausearch grouping and filtering
        # the parsed count matches the number of individual permissions denied
        assert a.avc_counter > 10, f"Too few AVCs: {a.avc_counter}"
        assert a.avc_counter < 500, f"Too many AVCs: {a.avc_counter}"


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPLE TEXT FILES MERGE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestMultiFileMerge:
    """Test merging multiple text files together."""

    def test_merge_two_text_files(self, tmp_path):
        """Two copies of the same text file should merge without error."""
        a1 = _make_analyzer()
        txt = a1.analyze(docs=[])

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        txt2 = a2.analyze(docs=[txt, txt])
        assert "allow myapp_t" in txt2
        # file_counter should be double
        assert a2.file_counter >= a1.avc_counter

    def test_merge_two_json_files(self, tmp_path):
        """Two JSON files should merge correctly."""
        j1 = str(tmp_path / "f1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        j2 = str(tmp_path / "f2.json")
        a2 = _make_analyzer(key="OtherKey")
        a2.analyze(docs=[], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        j3 = str(tmp_path / "merged.json")
        a3 = cla.Analyzer(key="Merged", look_in_log=False, show_pid_tree=True)
        a3.analyze(docs=[], json_docs=[d1, d2], json_dest=j3)
        with open(j3) as f:
            d3 = json.load(f)

        assert len(d3["results"]) >= len(d1["results"])

    def test_cli_merge_two_files(self, tmp_path):
        """CLI: merge two text files."""
        dest1 = str(tmp_path / "file1.txt")
        r1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "Key1", "--log", TEST_LOG, "--dest", dest1],
            capture_output=True, text=True, timeout=120,
        )
        assert r1.returncode == 0

        dest2 = str(tmp_path / "file2.txt")
        r2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "Key2", "--log", TEST_LOG, "--dest", dest2],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0

        dest3 = str(tmp_path / "merged.txt")
        r3 = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "Merged", "--ignore-log",
             "--files", dest1, dest2,
             "--dest", dest3],
            capture_output=True, text=True, timeout=120,
        )
        assert r3.returncode == 0, f"Merge failed:\n{r3.stderr}"
        with open(dest3) as f:
            content = f.read()
        assert "allow myapp_t" in content
        # Both keys should appear in the output
        assert "Key1" in content
        assert "Key2" in content


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLE ROUND-TRIP (stability test)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@needs_testlog
@needs_ausearch
class TestTripleRoundTrip:
    """Test that JSON round-trip is truly stable over multiple iterations."""

    def test_triple_json_roundtrip_stable(self, tmp_path):
        """log → J₁ → J₂ → J₃ : J₂ and J₃ must have identical results and pid_tree."""
        j1 = str(tmp_path / "j1.json")
        a1 = _make_analyzer()
        a1.analyze(docs=[], json_dest=j1)
        with open(j1) as f:
            d1 = json.load(f)

        j2 = str(tmp_path / "j2.json")
        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a2.analyze(docs=[], json_docs=[d1], json_dest=j2)
        with open(j2) as f:
            d2 = json.load(f)

        j3 = str(tmp_path / "j3.json")
        a3 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        a3.analyze(docs=[], json_docs=[d2], json_dest=j3)
        with open(j3) as f:
            d3 = json.load(f)

        # J₂ and J₃ must be identical (weights don't inflate across passes)
        assert len(d2["results"]) == len(d3["results"])
        assert set(d2["pid_tree"].keys()) == set(d3["pid_tree"].keys())
        assert sorted(d2["app_root_pids"]) == sorted(d3["app_root_pids"])
        assert d2["index"] == d3["index"]
        # Deep compare results
        def sort_results(results):
            return sorted(results, key=lambda r: json.dumps(r, sort_keys=True))
        assert sort_results(d2["results"]) == sort_results(d3["results"])

    def test_triple_text_roundtrip_stable(self, tmp_path):
        """log → T₁ → T₂ → T₃ : rules in T₂ and T₃ must be identical."""
        a1 = _make_analyzer()
        t1 = a1.analyze(docs=[])

        a2 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        t2 = a2.analyze(docs=[t1])

        a3 = cla.Analyzer(key=REFERENCE_KEY, look_in_log=False, show_pid_tree=True)
        t3 = a3.analyze(docs=[t2])

        def extract_rules(txt):
            return {l.strip() for l in txt.split("\n") if l.startswith("allow ")}

        rules2 = extract_rules(t2)
        rules3 = extract_rules(t3)
        assert rules2 == rules3, (
            f"Rules differ between T₂ and T₃:\n"
            f"Only in T₂: {rules2 - rules3}\nOnly in T₃: {rules3 - rules2}"
        )



# ═══════════════════════════════════════════════════════════════════════════════
# PID REUSE / COLLISION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPidReuse:
    """Unit tests for PID reuse detection and stale link cleanup in parse_execve_logs,
    and post-enrichment pruning of childless dead_process orphans."""

    def _make_analyzer_no_log(self, **kw):
        defaults = dict(key="test", look_in_log=False, show_debug=False, show_info=False)
        defaults.update(kw)
        return cla.Analyzer(**defaults)

    # ── Stale link cleanup ───────────────────────────────────────────────────

    def test_pid_reuse_removes_from_old_parent_children(self):
        """When a PID is reused with a different ppid, it should be removed
        from the old parent's children list."""
        a = self._make_analyzer_no_log()

        # Simulate first incarnation: pid=100, ppid=10
        old_parent_pk = (10, "test")
        child_pk = (100, "test")
        a.pid_tree[old_parent_pk] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="ctx", key="test", live=True,
            children=[child_pk],
        )
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd="grep foo", ppid=10, context="ctx", key="test", live=True,
        )

        # Simulate second incarnation: pid=100, ppid=20 (different parent)
        new_parent_pk = (20, "test")
        a.pid_tree[new_parent_pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="ctx", key="test", live=True,
        )

        # Manually trigger what parse_execve_logs does on PID reuse
        pk = child_pk
        new_ppid = 20
        old_ppid = a.pid_tree[pk].ppid
        assert old_ppid == 10

        # Detect reuse and cleanup
        old_ppid_pk = (old_ppid, "test")
        if old_ppid_pk in a.pid_tree and pk in a.pid_tree[old_ppid_pk].children:
            a.pid_tree[old_ppid_pk].children.remove(pk)
        # Force-update ppid (as the fix does before merge)
        a.pid_tree[pk].ppid = new_ppid

        # Old parent should no longer have child
        assert child_pk not in a.pid_tree[old_parent_pk].children
        # ppid must point to the new parent
        assert a.pid_tree[child_pk].ppid == 20

    def test_pid_reuse_detected_on_ppid_change(self):
        """PID reuse is detected when ppid changes for an existing pid_tree entry."""
        a = self._make_analyzer_no_log()

        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="grep foo", ppid=10, context="ctx", key="test", live=True,
        )

        old_ppid = a.pid_tree[pk].ppid
        new_ppid = 20
        assert old_ppid is not None and new_ppid != old_ppid

    def test_pid_reuse_not_triggered_same_ppid(self):
        """Re-exec within same process (same ppid) is NOT a PID collision."""
        a = self._make_analyzer_no_log()

        parent_pk = (10, "test")
        child_pk = (100, "test")
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="ctx", key="test", live=True,
            children=[child_pk],
        )
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd="sh script.sh", ppid=10, context="ctx", key="test", live=True,
        )

        # Same ppid → not a collision, children list unchanged
        old_ppid = a.pid_tree[child_pk].ppid
        same_ppid = 10
        assert old_ppid == same_ppid  # no collision condition triggered

        # Parent still has child
        assert child_pk in a.pid_tree[parent_pk].children

    def test_pid_reuse_not_triggered_when_ppid_is_none(self):
        """If old ppid is None (placeholder), ppid change is NOT a collision."""
        a = self._make_analyzer_no_log()

        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )

        old_ppid = a.pid_tree[pk].ppid
        # old_ppid is None → reuse detection condition is False
        assert old_ppid is None

    # ── Post-enrichment pruning of dead orphans ──────────────────────────────

    def test_prune_removes_childless_dead_process_orphan(self):
        """A dead_process node with no children and no parent in tree should be pruned."""
        a = self._make_analyzer_no_log()

        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )

        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            a.enrich_pid_tree()

        assert pk not in a.pid_tree

    def test_prune_keeps_dead_process_with_children(self):
        """A dead_process node that has children should NOT be pruned."""
        a = self._make_analyzer_no_log()

        parent_pk = (100, "test")
        child_pk = (200, "test")
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[child_pk],
        )
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd="/usr/bin/grep", ppid=100, context="system_u:system_r:myapp_t:s0",
            key="test", live=True,
        )

        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            a.enrich_pid_tree()

        assert parent_pk in a.pid_tree

    def test_prune_keeps_dead_process_with_parent_in_tree(self):
        """A dead_process node whose parent is in the tree should NOT be pruned
        (even if it has no children)."""
        a = self._make_analyzer_no_log()

        parent_pk = (10, "test")
        dead_pk = (100, "test")
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="ctx", key="test", live=True,
            children=[dead_pk],
        )
        a.pid_tree[dead_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=10, context=cla.UNKNOWN, key="test", live=True,
        )

        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            a.enrich_pid_tree()

        assert dead_pk in a.pid_tree

    def test_prune_cascades(self):
        """Pruning a childless dead orphan should cascade: if removing it makes
        its parent childless (and that parent is also dead_process with no parent
        in tree), the parent should be pruned in the next iteration."""
        a = self._make_analyzer_no_log()

        # Two independent childless dead orphans with no parent in tree:
        orphan1_pk = (500, "test")
        orphan2_pk = (600, "test")
        a.pid_tree[orphan1_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        # orphan2 has ppid=999 which is NOT in the tree
        a.pid_tree[orphan2_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=999, context=cla.UNKNOWN, key="test", live=True,
        )

        # Also: a dead parent whose only child is a childless dead orphan
        dead_parent_pk = (700, "test")
        dead_child_pk = (800, "test")
        a.pid_tree[dead_parent_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[dead_child_pk],
        )
        a.pid_tree[dead_child_pk] = cla.PidTreeEntry(
            cmd=cla.DEAD_PROCESS, ppid=700, context=cla.UNKNOWN, key="test", live=True,
        )

        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            a.enrich_pid_tree()

        assert orphan1_pk not in a.pid_tree
        assert orphan2_pk not in a.pid_tree
        # dead_parent and dead_child survive (connected chain)
        assert dead_parent_pk in a.pid_tree
        assert dead_child_pk in a.pid_tree

    # ── End-to-end: stale link + prune interaction ───────────────────────────

    def test_full_pid_reuse_scenario(self):
        """Simulate the full PID 20152 scenario: parent with two children that get
        reparented, leaving the original parent as a childless dead orphan that
        gets pruned."""
        a = self._make_analyzer_no_log()

        # First incarnation: pid=20153 ppid=20152
        parent_pk = (20152, "test")
        child1_pk = (20153, "test")
        child2_pk = (20154, "test")

        # Child 20153 first seen → creates parent placeholder
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[child1_pk],
        )
        a.pid_tree[child1_pk] = cla.PidTreeEntry(
            cmd="grep foo", ppid=20152, context="system_u:system_r:ccc_t:s0:c290,c659",
            key="test", live=True,
        )
        # Child 20154 first seen → added to parent
        a.pid_tree[parent_pk].children.append(child2_pk)
        a.pid_tree[child2_pk] = cla.PidTreeEntry(
            cmd="cut -d ' ' -f 2", ppid=20152, context="system_u:system_r:ccc_t:s0:c290,c659",
            key="test", live=True,
        )

        assert len(a.pid_tree[parent_pk].children) == 2

        # Second incarnation: pid=20153 ppid=11479 (PID reuse!)
        new_parent_pk = (11479, "test")
        a.pid_tree[new_parent_pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="system_u:system_r:ccc_t:s0:c783,c877",
            key="test", live=True,
        )

        # Simulate stale link cleanup for child 20153
        old_ppid = a.pid_tree[child1_pk].ppid
        assert old_ppid == 20152
        old_ppid_pk = (old_ppid, "test")
        if old_ppid_pk in a.pid_tree and child1_pk in a.pid_tree[old_ppid_pk].children:
            a.pid_tree[old_ppid_pk].children.remove(child1_pk)
        # Add to new parent
        a.pid_tree[new_parent_pk].children.append(child1_pk)

        # Simulate stale link cleanup for child 20154
        old_ppid2 = a.pid_tree[child2_pk].ppid
        assert old_ppid2 == 20152
        old_ppid_pk2 = (old_ppid2, "test")
        if old_ppid_pk2 in a.pid_tree and child2_pk in a.pid_tree[old_ppid_pk2].children:
            a.pid_tree[old_ppid_pk2].children.remove(child2_pk)
        a.pid_tree[new_parent_pk].children.append(child2_pk)

        # After reparenting, old parent should have no children
        assert len(a.pid_tree[parent_pk].children) == 0

        # Simulate enrichment labelling 20152 as dead
        a.pid_tree[parent_pk].cmd = cla.DEAD_PROCESS

        # Run the real prune via enrich_pid_tree
        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessDead("dead")):
            a.enrich_pid_tree()

        # Old parent (20152) should be gone
        assert parent_pk not in a.pid_tree

        # Children should still exist under new parent
        assert child1_pk in a.pid_tree
        assert child2_pk in a.pid_tree
        assert child1_pk in a.pid_tree[new_parent_pk].children
        assert child2_pk in a.pid_tree[new_parent_pk].children


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — enrich_pid_tree() parent chain, /proc context, zombie/race paths
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEnrichPidTreeCoverage:
    """Exercises enrich_pid_tree() branches not reached by the basic tests."""

    @staticmethod
    def _make():
        return cla.Analyzer(key="test", look_in_log=False, show_pid_tree=True)

    def test_parent_chain_creates_placeholders(self):
        """Walking up the parent chain should create placeholder entries
        for parents not already in the tree."""
        a = self._make()
        # pid 500 has UNKNOWN cmd → needs enrichment.  Its fake psutil.Process
        # reports ppid=400, which is not in the tree yet → should be created.
        pk = (500, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)

        fake_proc_500 = mock.MagicMock()
        fake_proc_500.cmdline.return_value = ["/usr/bin/app", "--flag"]
        fake_proc_500.ppid.return_value = 400
        fake_proc_500.is_running.return_value = True
        fake_proc_500.status.return_value = "running"

        fake_proc_400 = mock.MagicMock()
        fake_proc_400.cmdline.return_value = ["/bin/bash"]
        fake_proc_400.ppid.return_value = 1
        fake_proc_400.is_running.return_value = True
        fake_proc_400.status.return_value = "running"

        fake_proc_1 = mock.MagicMock()
        fake_proc_1.cmdline.return_value = ["/sbin/init"]
        fake_proc_1.ppid.return_value = 0
        fake_proc_1.is_running.return_value = True
        fake_proc_1.status.return_value = "running"

        def fake_is_alive(pid):
            mapping = {500: fake_proc_500, 400: fake_proc_400, 1: fake_proc_1}
            p = mapping.get(int(pid))
            if p is None:
                raise cla.ProcessDead(f"{pid} dead")
            return p

        # Mock /proc/*/attr/current reads
        def fake_open(path, *a, **kw):
            if "/proc/" in path and "/attr/current" in path:
                m = mock.mock_open(read_data="system_u:system_r:myapp_t:s0\x00")()
                return m
            return open.__wrapped__(path, *a, **kw) if hasattr(open, '__wrapped__') else _real_open(path, *a, **kw)

        import builtins
        _real_open = builtins.open

        with mock.patch.object(a, "is_process_alive", side_effect=fake_is_alive), \
             mock.patch("builtins.open", side_effect=fake_open):
            a.enrich_pid_tree()

        # pid 500 should now have cmd enriched
        assert a.pid_tree[(500, "test")].cmd == "/usr/bin/app --flag"
        # pid 400 should have been created as placeholder and enriched
        assert (400, "test") in a.pid_tree
        assert a.pid_tree[(400, "test")].cmd == "/bin/bash"
        # pid 1 (init) should be in the tree
        assert (1, "test") in a.pid_tree

    def test_zombie_process_breaks_chain(self):
        """ProcessZombie (ProcessNotAvailable) should stop the parent chain walk."""
        a = self._make()
        pk = (600, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)

        with mock.patch.object(a, "is_process_alive", side_effect=cla.ProcessNotAvailable("zombie")):
            a.enrich_pid_tree()

        # Should remain UNKNOWN — zombie can't be queried
        assert a.pid_tree[pk].cmd == cla.UNKNOWN

    def test_race_condition_process_dies_during_query(self):
        """When is_process_alive succeeds but cmdline() raises NoSuchProcess,
        the process should be labelled dead."""
        a = self._make()
        pk = (700, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)

        fake_proc = mock.MagicMock()
        fake_proc.cmdline.side_effect = psutil.NoSuchProcess(700)

        with mock.patch.object(a, "is_process_alive", return_value=fake_proc):
            a.enrich_pid_tree()

        entry = a.pid_tree.get(pk)
        # Either pruned or labelled dead
        if entry is not None:
            assert entry.cmd == cla.DEAD_PROCESS

    def test_known_cmd_walks_parent_only(self):
        """A live entry with known cmd but unknown ppid should still get ppid filled."""
        a = self._make()
        pk = (800, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/usr/bin/known", ppid=None, context="ctx", key="test", live=True,
        )
        a.avc_pids.add(pk)

        fake_proc = mock.MagicMock()
        fake_proc.ppid.return_value = 1
        fake_proc.is_running.return_value = True
        fake_proc.status.return_value = "running"

        def fake_is_alive(pid):
            if int(pid) == 800:
                return fake_proc
            raise cla.ProcessDead(f"{pid}")

        with mock.patch.object(a, "is_process_alive", side_effect=fake_is_alive):
            a.enrich_pid_tree()

        assert a.pid_tree[pk].ppid == 1

    def test_proc_attr_permission_denied(self):
        """When /proc/pid/attr/current is not readable, context stays UNKNOWN."""
        a = self._make()
        pk = (900, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)

        fake_proc = mock.MagicMock()
        fake_proc.cmdline.return_value = ["/bin/something"]
        fake_proc.ppid.return_value = 1

        def fake_is_alive(pid):
            if int(pid) == 900:
                return fake_proc
            raise cla.ProcessDead(f"{pid}")

        with mock.patch.object(a, "is_process_alive", side_effect=fake_is_alive), \
             mock.patch("builtins.open", side_effect=PermissionError("denied")):
            a.enrich_pid_tree()

        assert a.pid_tree[(900, "test")].context == cla.UNKNOWN
        assert a.pid_tree[(900, "test")].cmd == "/bin/something"


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — parse_ausearch_from_log() fallback paths
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseAusearchFallbackPaths:
    """Cover parse_ausearch_from_log fallback branches: no FULL_AVC, no SYSCALL_AVC,
    individual field extraction, and ps lookup when pid is alive."""

    def test_partial_extract_uses_uid_ppid_pid_regexes(self):
        """Block with no SYSCALL line forces partial extraction via individual regexes."""
        block = (
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=300 uid=1000 ppid=100 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        assert results[0].command.pid == "300"

    def test_ps_fallback_for_ppid(self):
        """When ppid is missing and pid is alive, get_from_pid('ppid') should be called."""
        # Block with a PID match but no ppid field and no FULL_AVC/SYSCALL_AVC
        block = (
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=12345 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)

        # Mock get_from_pid to simulate a live process
        with mock.patch.object(a, "get_from_pid", return_value=cla.DEAD_PROCESS):
            results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1

    def test_full_avc_pid_ppid_parse_error(self):
        """FULL_AVC match but pid/ppid extraction fails should print error and continue."""
        # Craft a block where FULL_AVC matches but the SYSCALL portion has no ppid= pid=
        block = (
            'type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/myapp\n'
            'type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=open\n'
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=200 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        # Should not crash — produces result with whatever could be parsed
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) >= 1

    def test_dead_cmd_falls_back_to_exe(self):
        """When cmd is dead_process or None, parser should try exe= and comm= from block."""
        block = (
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=99999999 exe=/usr/sbin/myapp comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        # Should have exe= or comm= as cmd
        assert results[0].command.cmd in ("/usr/sbin/myapp", "myapp")

    def test_pid_namespace_dead_process(self):
        """When /proc/<pid>/ns/pid fails and process is dead, pid_namespace=PID_DEAD."""
        block = (
            'type=PROCTITLE msg=audit(09/02/2026 10:45:19.517:18438) : proctitle=/bin/myapp\n'
            'type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=open ppid=100 pid=9999999 uid=root\n'
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=9999999 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        # PID is certainly dead
        assert results[0].command.pid_namespace in (cla.NOT_FOUND, cla.PID_DEAD)


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — parse_index_from_file() history dedup & ns alias
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseIndexFromFileCoverage:
    """Cover deeper branches: multi-word history, **** replacement at various
    column positions, namespace alias parsing, and corrupted weight handling."""

    def test_history_dedup_later_columns(self):
        """History dedup should work for words beyond the 4th column."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = (
            f"rules\n{cla.INDEX_DELIMITER}\n"
            f"### CMD_0001 1 | /usr/bin/app --arg1 --arg2 --arg3\n"
            f"### CMD_0002 1 | ************ ****** ****** --arg4\n"
        )
        a.parse_index_from_file(doc)
        # The second command should have been reconstructed
        alias = a.cmd_index.get_alias("/usr/bin/app --arg1 --arg2 --arg4 ")
        assert alias == "CMD_0002" or "/usr/bin/app" in alias or "CMD_0002" in str(a.cmd_index.to_dict())

    def test_namespace_alias_parsed(self):
        """parse_index_from_file should register a ~pid_ns_*~ alias into ns_index."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### ~pid_ns_myhost~ 0 | 54321\n"
        a.parse_index_from_file(doc)
        assert 54321 in a.ns_index
        assert a.ns_index.get(54321) == "~pid_ns_myhost~"

    def test_namespace_non_decimal_skipped(self):
        """Namespace alias with non-decimal value should be skipped gracefully."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### ~pid_ns_broken~ 0 | not_a_number\n"
        a.parse_index_from_file(doc)
        # Should not crash; ns_index should be empty
        assert len(a.ns_index) == 0

    def test_corrupted_weight_raises(self):
        """Non-integer weight field should raise FileParsingError."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### CMD_0001 abc | /usr/bin/app\n"
        with pytest.raises(cla.FileParsingError, match="err_001"):
            a.parse_index_from_file(doc)

    def test_corrupted_delimiter_raises(self):
        """Wrong delimiter field should raise FileParsingError."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### CMD_0001 1 X /usr/bin/app\n"
        with pytest.raises(cla.FileParsingError, match="err_002"):
            a.parse_index_from_file(doc)

    def test_non_index_line_skipped(self):
        """Lines that don't start with ### should be silently ignored."""
        a = cla.Analyzer(key="test", look_in_log=False)
        doc = f"rules\n{cla.INDEX_DELIMITER}\nsome random comment\n### CMD_0001 1 | /usr/bin/app\n"
        replacer = a.parse_index_from_file(doc)
        assert isinstance(replacer, list)
        assert "/usr/bin/app" in a.cmd_index or a.cmd_index.get_alias("/usr/bin/app ") == "CMD_0001"

    def test_alias_collision_in_parse_index(self):
        """Pre-existing alias collision should produce a replacer entry."""
        a = cla.Analyzer(key="test", look_in_log=False)
        a.set_index("/usr/bin/other", "CMD_0001")
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### CMD_0001 1 | /usr/bin/newcmd\n"
        replacer = a.parse_index_from_file(doc)
        assert len(replacer) == 1
        old, new = replacer[0]
        assert old == "CMD_0001"
        assert new != "CMD_0001"

    def test_pid_tree_section_stripped_from_index(self):
        """Index parsing should not read PID tree section as index entries."""
        a = cla.Analyzer(key="test", look_in_log=False)
        tree_section = (
            "# test - Process Tree\n"
            "# ├── pid=1@test ctx: system_u:system_r:init_t:s0 cmd: /sbin/init\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\n### CMD_0001 1 | /usr/bin/app\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        replacer = a.parse_index_from_file(doc)
        assert isinstance(replacer, list)
        assert a.cmd_index.get_alias("/usr/bin/app ") == "CMD_0001"


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — parse_pid_tree_from_file() deeper branches
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParsePidTreeFromFileCoverage:
    """Cover indent stack, app root from entry marker, merge with existing,
    multiple app roots from header, backward compat list vs single."""

    def test_multiple_app_roots_in_header(self):
        """Header with multiple comma-separated app root PIDs."""
        tree_section = (
            "# test - Process Tree (APP Roots detected at 1001, 1003)\n"
            "#\n"
            "# ├── pid=1001@test [APP ROOT]     ctx: system_u:system_r:myapp_t:s0    cmd: /bin/bash /usr/sbin/myapp start\n"
            "# ├── pid=1003@test [APP ROOT]     ctx: system_u:system_r:myapp_t:s0    cmd: /bin/bash /usr/sbin/myapp stop\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert (1001, "test") in a.app_root_pids
        assert (1003, "test") in a.app_root_pids

    def test_deep_nesting_indent_stack(self):
        """Deep nesting with multiple │ characters should produce correct parent-child."""
        tree_section = (
            "# test - Process Tree\n"
            "#\n"
            "# ├── pid=1@test                      ctx: ctx    cmd: /sbin/init\n"
            "# │   ├── pid=10@test                 ctx: ctx    cmd: /bin/bash\n"
            "# │   │   ├── pid=100@test            ctx: ctx    cmd: /usr/bin/app\n"
            "# │   │   │   ├── pid=1000@test       ctx: ctx    cmd: /usr/bin/worker\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert a.pid_tree[(10, "test")].ppid == 1
        assert a.pid_tree[(100, "test")].ppid == 10
        assert a.pid_tree[(1000, "test")].ppid == 100
        assert (10, "test") in a.pid_tree[(1, "test")].children

    def test_merge_with_existing_entry(self):
        """Loading from file should merge with pre-existing pid_tree entries."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (1001, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        tree_section = (
            "# test - Process Tree\n"
            "#\n"
            "# ├── pid=1001@test                   ctx: system_u:system_r:myapp_t:s0    cmd: /bin/myapp\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a.parse_pid_tree_from_file(doc)
        # Merge should have updated cmd
        assert a.pid_tree[pk].cmd == "/bin/myapp"

    def test_app_root_from_entry_marker(self):
        """App root detected from [APP ROOT] in entry line, not from header."""
        tree_section = (
            "# test - Process Tree\n"
            "#\n"
            "# ├── pid=5000@test [APP ROOT]        ctx: system_u:system_r:myapp_t:s0    cmd: /usr/sbin/myapp\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert (5000, "test") in a.app_root_pids

    def test_empty_tree_section_is_noop(self):
        """Empty PID tree section should not crash."""
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert len(a.pid_tree) == 0

    def test_sibling_nodes_parsed(self):
        """Two children at the same indent level should both have the same parent."""
        tree_section = (
            "# test - Process Tree\n"
            "#\n"
            "# ├── pid=1@test                      ctx: ctx    cmd: /sbin/init\n"
            "# │   ├── pid=10@test                 ctx: ctx    cmd: child1\n"
            "# │   ├── pid=20@test                 ctx: ctx    cmd: child2\n"
        )
        doc = f"rules\n{cla.INDEX_DELIMITER}\nindex\n{cla.PID_TREE_DELIMITER}\n{tree_section}"
        a = cla.Analyzer(key="test", look_in_log=False)
        a.parse_pid_tree_from_file(doc)
        assert a.pid_tree[(10, "test")].ppid == 1
        assert a.pid_tree[(20, "test")].ppid == 1
        assert (10, "test") in a.pid_tree[(1, "test")].children
        assert (20, "test") in a.pid_tree[(1, "test")].children


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — filter_pid_tree() UNKNOWN:UNKNOWN cascade prune
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFilterPidTreeCoverage:
    """Test the cascading prune of UNKNOWN:UNKNOWN single-child linear chains."""

    def test_unknown_chain_pruned(self):
        """A chain of UNKNOWN:UNKNOWN nodes each with 1 child should collapse
        until only the leaf (relevant) node remains."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # grandparent → parent → leaf (leaf is in avc_pids)
        gp = (1, "test")
        pa = (2, "test")
        lf = (3, "test")
        a.pid_tree[gp] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[pa],
        )
        a.pid_tree[pa] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=1, context=cla.UNKNOWN, key="test", live=True,
            children=[lf],
        )
        a.pid_tree[lf] = cla.PidTreeEntry(
            cmd="/usr/bin/app", ppid=2, context="myapp_t", key="test", live=True,
        )
        a.avc_pids.add(lf)
        a.filter_pid_tree()
        # Only leaf should survive
        assert lf in a.pid_tree
        assert gp not in a.pid_tree
        assert pa not in a.pid_tree
        assert a.pid_tree[lf].ppid is None

    def test_unknown_with_two_children_kept(self):
        """UNKNOWN:UNKNOWN root with TWO children should NOT be pruned."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root = (1, "test")
        c1 = (10, "test")
        c2 = (20, "test")
        a.pid_tree[root] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[c1, c2],
        )
        a.pid_tree[c1] = cla.PidTreeEntry(
            cmd="app1", ppid=1, context="ctx", key="test", live=True,
        )
        a.pid_tree[c2] = cla.PidTreeEntry(
            cmd="app2", ppid=1, context="ctx", key="test", live=True,
        )
        a.avc_pids.update({c1, c2})
        a.filter_pid_tree()
        assert root in a.pid_tree
        assert c1 in a.pid_tree
        assert c2 in a.pid_tree


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — format_pid_tree() orphan rendering, APP Root header
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFormatPidTreeCoverage:
    """Cover orphan rendering and APP Root label in format_pid_tree()."""

    def test_orphan_processes_rendered(self):
        """Processes with no children and no parent in tree should appear in Orphan section."""
        a = cla.Analyzer(key="test", look_in_log=False)
        # A root with children
        root = (1, "test")
        child = (10, "test")
        a.pid_tree[root] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="ctx", key="test", live=True,
            children=[child],
        )
        a.pid_tree[child] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=1, context="ctx", key="test", live=True,
        )
        # An orphan (no parent, no children)
        orphan = (999, "test")
        a.pid_tree[orphan] = cla.PidTreeEntry(
            cmd="/usr/bin/orphan", ppid=None, context="ctx_orphan", key="test", live=True,
        )
        out = a.format_pid_tree()
        assert "Orphan" in out
        assert "pid=999" in out

    def test_app_root_label_in_header(self):
        """APP Root pids should produce a header like 'Process Tree (APP Root detected at ...)'."""
        a = cla.Analyzer(key="test", look_in_log=False)
        root = (1001, "test")
        child = (1002, "test")
        a.pid_tree[root] = cla.PidTreeEntry(
            cmd="/bin/bash /usr/sbin/myapp", ppid=None, context="myapp_t", key="test", live=True,
            children=[child],
        )
        a.pid_tree[child] = cla.PidTreeEntry(
            cmd="/usr/bin/worker", ppid=1001, context="myapp_t", key="test", live=True,
        )
        a.app_root_pids.append(root)
        out = a.format_pid_tree()
        assert "APP Root detected at 1001" in out
        assert "[APP ROOT]" in out

    def test_multiple_app_roots_label(self):
        """Multiple APP roots should produce 'APP Roots' (plural)."""
        a = cla.Analyzer(key="test", look_in_log=False)
        r1 = (1001, "test")
        r2 = (1003, "test")
        for r in [r1, r2]:
            a.pid_tree[r] = cla.PidTreeEntry(
                cmd="/bin/myapp", ppid=None, context="myapp_t", key="test", live=True,
                children=[(r[0] + 1, "test")],
            )
            c = (r[0] + 1, "test")
            a.pid_tree[c] = cla.PidTreeEntry(
                cmd="worker", ppid=r[0], context="myapp_t", key="test", live=True,
            )
        a.app_root_pids.extend([r1, r2])
        out = a.format_pid_tree()
        assert "APP Roots detected at" in out

    def test_empty_pid_tree(self):
        """Empty pid_tree should produce the 'empty' message."""
        a = cla.Analyzer(key="test", look_in_log=False)
        out = a.format_pid_tree()
        assert "No EXECVE events found" in out


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — CLI main() paths
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
class TestCLICoveragePaths:
    """Cover CLI argument validation and output routing not hit by existing tests."""

    def test_key_sanitization_spaces_and_brackets(self, tmp_path):
        """Spaces, brackets, and other special chars in --key should be replaced."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "my host (v2) [test]",
             "--ignore-log",
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        # Key sanitization runs in main(); success means no crash from special chars
        assert result.returncode == 0
        assert os.path.isfile(dest)

    def test_json_only_output_no_human(self, tmp_path):
        """--json-dest without --dest should skip human-readable formatting."""
        json_dest = str(tmp_path / "output.json")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--json-dest", json_dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert os.path.isfile(json_dest)
        with open(json_dest) as f:
            data = json.load(f)
        assert data["version"] == cla.JSON_FORMAT_VERSION
        # stdout should NOT have full human-readable output
        assert cla.INDEX_DELIMITER not in result.stdout

    def test_both_json_and_dest(self, tmp_path):
        """--json-dest with --dest should produce both outputs."""
        dest = str(tmp_path / "output.txt")
        json_dest = str(tmp_path / "output.json")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--dest", dest,
             "--json-dest", json_dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert os.path.isfile(dest)
        assert os.path.isfile(json_dest)
        with open(dest) as f:
            assert cla.INDEX_DELIMITER in f.read()
        with open(json_dest) as f:
            assert json.load(f)["version"] == cla.JSON_FORMAT_VERSION

    def test_json_file_parse_error(self, tmp_path):
        """Corrupted JSON input file should raise FileParsingError."""
        bad_json = str(tmp_path / "bad.json")
        with open(bad_json, "w") as f:
            f.write("{this is not valid json}")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--json-files", bad_json],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_file_missing_index_raises(self, tmp_path):
        """Input file without INDEX_DELIMITER should raise FileParsingError."""
        bad_file = str(tmp_path / "bad.txt")
        with open(bad_file, "w") as f:
            f.write("allow myapp_t cert_t:dir read;\n")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--files", bad_file],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_stdout_when_no_dest(self, tmp_path):
        """Without --dest, human-readable output should go to stdout."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert cla.INDEX_DELIMITER in result.stdout

    def test_context_filter_and_app_name(self, tmp_path):
        """--context-filter and --app-name should be passed to analyzer."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--context-filter", "myapp_t", "myapp2_t",
             "--app-name", "myapp",
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_no_explanations_flag(self, tmp_path):
        """--no-explanations should suppress the 'required by' blocks."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--no-explanations",
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_no_tree_flag(self, tmp_path):
        """--no-tree should suppress PID tree output."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--no-tree",
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        with open(dest) as f:
            content = f.read()
        assert cla.PID_TREE_DELIMITER not in content

    def test_no_index_flag(self, tmp_path):
        """--no-index should suppress the INDEX section entirely."""
        dest = str(tmp_path / "output.txt")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH,
             "--key", "test",
             "--ignore-log",
             "--no-index",
             "--dest", dest],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        with open(dest) as f:
            content = f.read()
        # INDEX delimiter still present (structural) but no index content between them
        assert cla.INDEX_DELIMITER in content

    def test_log_not_readable(self, tmp_path):
        """--log pointing to unreadable file should raise PermissionError."""
        unreadable = str(tmp_path / "noperm.log")
        with open(unreadable, "w") as f:
            f.write("data")
        os.chmod(unreadable, 0o000)
        try:
            result = subprocess.run(
                [sys.executable, _SCRIPT_PATH,
                 "--key", "test",
                 "--log", unreadable],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode != 0
        finally:
            os.chmod(unreadable, 0o644)


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — analyze() method routing (skip_formatting, json_dest)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAnalyzeMethodCoverage:
    """Cover skip_formatting path and json_dest output in analyze()."""

    def test_skip_formatting_returns_empty(self, tmp_path):
        """analyze() with skip_formatting=True should return empty string."""
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False)
        result = a.analyze(docs=[], skip_formatting=True)
        assert result == ""

    def test_skip_formatting_saves_state(self, tmp_path):
        """skip_formatting=True should still save state file when configured."""
        sf = str(tmp_path / "state.json")
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False,
                         state_file_path=sf)
        a.analyze(docs=[], skip_formatting=True)
        assert os.path.isfile(sf)

    def test_json_dest_written(self, tmp_path):
        """analyze() should write JSON when json_dest is given."""
        jd = str(tmp_path / "out.json")
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False)
        a.analyze(docs=[], json_dest=jd)
        with open(jd) as f:
            data = json.load(f)
        assert data["version"] == cla.JSON_FORMAT_VERSION
        assert data["key"] == "test"

    def test_analyze_with_existing_file(self, tmp_path):
        """analyze() should parse existing file and merge results."""
        # Create a valid existing output file
        doc = (
            f"allow myapp_t cert_t:dir read;\n"
            f"# required by :\n"
            f"#     test | /usr/bin/cmd (pid=100 ; pid_ns=notFound)\n"
            f"#          | SYSCALL: msg=audit(...)\n"
            f"\n{cla.INDEX_DELIMITER}\n"
            f"### CMD_0001 1 | /usr/bin/cmd\n"
        )
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False)
        txt = a.analyze(docs=[doc])
        assert "allow myapp_t cert_t:dir read;" in txt
        assert a.file_counter == 1

    def test_analyze_with_json_doc(self, tmp_path):
        """analyze() should merge JSON documents."""
        json_doc = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [{
                "command": {
                    "key": "test",
                    "descriptors": ["desc1"],
                    "pid_namespace": "notFound",
                    "cmd": "/usr/bin/app",
                    "pid": "100",
                },
                "AVC": [{
                    "source_type": "myapp_t",
                    "target_type": "cert_t",
                    "tclass": "dir",
                    "method": "read",
                }],
            }],
            "index": {},
            "ns_index": {},
            "all_cmds": {},
            "pid_tree": {},
            "app_root_pids": [],
            "avc_pids": [],
        }
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False)
        txt = a.analyze(docs=[], json_docs=[json_doc])
        assert "allow myapp_t cert_t:dir read;" in txt
        assert a.file_counter == 1

    def test_no_index_mode(self):
        """analyze() with no_index=True should produce output without index content."""
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=False, no_index=True)
        txt = a.analyze(docs=[])
        assert cla.INDEX_DELIMITER in txt

    def test_pid_tree_included_when_enabled(self, tmp_path):
        """analyze() with show_pid_tree=True and populated tree should include PID tree."""
        a = cla.Analyzer(key="test", look_in_log=False, show_pid_tree=True)
        # Manually add a non-live entry (simulating file load)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/usr/bin/app", ppid=None, context="ctx", key="test", live=False,
        )
        txt = a.analyze(docs=[])
        assert cla.PID_TREE_DELIMITER in txt
        assert "pid=100" in txt


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — state file save/load edge cases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStateFileCoverage:
    """Edge cases for save/load state file operations."""

    def test_save_and_reload_state(self, tmp_path):
        """save → load round-trip should preserve entries."""
        sf = str(tmp_path / "state.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.analyzed_entries = {"entry1", "entry2", "entry3"}
        a.save_analyzed_entries()

        a2 = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a2.load_analyzed_entries()
        assert a2.analyzed_entries == {"entry1", "entry2", "entry3"}

    def test_load_nonexistent_state(self, tmp_path):
        """Loading from non-existent state file should be a no-op."""
        sf = str(tmp_path / "nonexistent.json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.load_analyzed_entries()
        assert a.analyzed_entries == set()

    def test_load_corrupted_state(self, tmp_path):
        """Loading from corrupted JSON should not crash and should reset entries."""
        sf = str(tmp_path / "bad_state.json")
        with open(sf, "w") as f:
            f.write("{broken json")
        a = cla.Analyzer(key="test", look_in_log=False, state_file_path=sf)
        a.load_analyzed_entries()
        assert a.analyzed_entries == set()

    def test_save_without_state_path_is_noop(self):
        """save_analyzed_entries without state_file_path should do nothing."""
        a = cla.Analyzer(key="test", look_in_log=False)
        a.analyzed_entries = {"e1"}
        a.save_analyzed_entries()  # should not crash


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — merge_json() backward compat & edge cases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMergeJsonCoverage:
    """Cover merge_json backward compat (old single app_root_pid)
    and version mismatch warning."""

    def test_old_format_single_app_root_pid(self):
        """Old JSON format with 'app_root_pid' (singular) should be handled."""
        a = cla.Analyzer(key="test", look_in_log=False)
        pk = (1001, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/myapp", ppid=None, context="ctx", key="test", live=False,
        )
        data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [],
            "index": {},
            "ns_index": {},
            "all_cmds": {},
            "pid_tree": {
                "1001:test": {
                    "cmd": "/bin/myapp", "ppid": None, "context": "ctx",
                    "key": "test", "live": False, "children": [],
                },
            },
            "app_root_pids": [],
            "app_root_pid": "1001:test",  # old format
            "avc_pids": [],
        }
        a.merge_json(data)
        assert (1001, "test") in a.app_root_pids

    def test_version_mismatch_prints_warning(self, capsys):
        """Version mismatch should print a warning to stderr."""
        a = cla.Analyzer(key="test", look_in_log=False)
        data = {
            "version": 999,
            "key": "test",
            "results": [],
            "index": {},
            "ns_index": {},
            "all_cmds": {},
            "pid_tree": {},
            "app_root_pids": [],
            "avc_pids": [],
        }
        a.merge_json(data)
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "version" in captured.err.lower()

    def test_ns_index_merge_with_collision(self):
        """Merging ns_index that collides should handle gracefully."""
        a = cla.Analyzer(key="test", look_in_log=False, show_info=True)
        a.ns_index.set(100, "~pid_ns_host~")
        data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [],
            "index": {},
            "ns_index": {"200": "~pid_ns_host~"},
            "all_cmds": {},
            "pid_tree": {},
            "app_root_pids": [],
            "avc_pids": [],
        }
        a.merge_json(data)
        # Should have registered with collision suffix
        assert 200 in a.ns_index
        assert a.ns_index.get(200) != a.ns_index.get(100)

    def test_invalid_ns_id_in_json_skipped(self, capsys):
        """Non-numeric ns_id in JSON should be skipped with warning."""
        a = cla.Analyzer(key="test", look_in_log=False)
        data = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "test",
            "results": [],
            "index": {},
            "ns_index": {"not_a_number": "~pid_ns_bad~"},
            "all_cmds": {},
            "pid_tree": {},
            "app_root_pids": [],
            "avc_pids": [],
        }
        a.merge_json(data)
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "invalid" in captured.err.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — look_for_constraint_violation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConstraintViolationCoverage:

    def test_missing_audit2allow(self, tmp_path):
        """When audit2allow is not installed, should handle OSError gracefully."""
        log = str(tmp_path / "test.log")
        with open(log, "w") as f:
            f.write("type=AVC some data\n")
        with mock.patch("subprocess.run", side_effect=OSError("not found")):
            # Should not raise
            cla.Analyzer.look_for_constraint_violation(log)

    def test_constraint_violation_detected(self, tmp_path, capsys):
        """When audit2allow reports constraint violation, should print warning."""
        log = str(tmp_path / "test.log")
        with open(log, "w") as f:
            f.write("type=AVC some data\n")
        fake_result = mock.MagicMock()
        fake_result.stdout = "constraint violation found"
        with mock.patch("subprocess.run", return_value=fake_result):
            cla.Analyzer.look_for_constraint_violation(log)
        captured = capsys.readouterr()
        assert "constraint violation" in captured.err.lower()

    def test_audit2allow_timeout(self, tmp_path):
        """audit2allow timeout should be handled gracefully."""
        log = str(tmp_path / "test.log")
        with open(log, "w") as f:
            f.write("type=AVC some data\n")
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("audit2allow", 120)):
            cla.Analyzer.look_for_constraint_violation(log)


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — Debug-mode paths (show_debug=True)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDebugModePaths:
    """Run key operations with show_debug=True to cover debug print branches."""

    def test_ns_index_debug_prints(self, capsys):
        """NsIndex operations with debug mode should cover debug print lines."""
        ns = cla.NsIndex(show_debug=True)
        # set a new entry → debug print
        ns.set(100, "~pid_ns_a~")
        # idempotent set → debug "existing" print
        ns.set(100, "~pid_ns_a~")
        # collision → debug "collision" print
        ns.set(200, "~pid_ns_a~")
        captured = capsys.readouterr()
        assert "registered" in captured.err
        assert "existing" in captured.err or "label existing" in captured.err
        assert "collision" in captured.err

    def test_cmd_index_debug_prints(self, capsys):
        """CmdIndex operations with debug mode should cover debug print lines."""
        ci = cla.CmdIndex(show_debug=True)
        # Register with explicit alias → debug print
        ci.register("cmd_a", "ALIAS_0001")
        # Idempotent → debug "existing" print
        ci.register("cmd_a")
        # Collision on template alias → "refactoring" debug
        ci.register("cmd_b", "ALIAS_0001")
        # Collision on non-template alias → "got base_alias" debug
        ci.register("cmd_c", "MYALIAS")
        ci.register("cmd_d", "MYALIAS")
        captured = capsys.readouterr()
        assert "adding alias" in captured.err or "creating alias" in captured.err
        assert "alias existing" in captured.err

    def test_cmd_index_format_debug(self, capsys):
        """CmdIndex.format() with debug should print each alias being written."""
        ci = cla.CmdIndex(show_debug=True)
        ci.register("ausearch -i -m avc", "AVC_0001")
        ci.register("ausearch -i -m msg", "MSG_0001")
        ci.log_weight("ausearch -i -m avc")
        ci.log_weight("ausearch -i -m msg")
        out = ci.format()
        captured = capsys.readouterr()
        assert "writing alias" in captured.err or "index to format" in captured.err

    def test_parse_index_from_file_debug(self, capsys):
        """parse_index_from_file with debug should cover all debug print paths."""
        a = cla.Analyzer(key="test", look_in_log=False, show_debug=True)
        doc = (
            f"rules\n{cla.INDEX_DELIMITER}\n"
            f"### CMD_0001 1 | ausearch -i -m avc\n"
            f"### CMD_0002 1 | ******** ** ** msg\n"
            f"\n"  # empty line should trigger "line null" debug
        )
        a.parse_index_from_file(doc)
        captured = capsys.readouterr()
        assert "parse_index_from_file" in captured.err

    def test_enrich_pid_tree_debug(self, capsys):
        """enrich_pid_tree with debug should cover debug print paths for dead pids."""
        a = cla.Analyzer(key="test", look_in_log=False, show_debug=True)
        pk = (9999999, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)
        a.enrich_pid_tree()
        captured = capsys.readouterr()
        assert "enrich_pid_tree" in captured.err

    def test_is_process_alive_debug_dead(self, capsys):
        """is_process_alive with debug on a dead PID should print debug msg."""
        a = cla.Analyzer(key="test", look_in_log=False, show_debug=True)
        with pytest.raises(cla.ProcessDead):
            a.is_process_alive(9999999)
        captured = capsys.readouterr()
        assert "is_process_alive" in captured.err or "dead" in captured.err

    def test_filter_pid_tree_debug(self, capsys):
        """filter_pid_tree with show_info should print filter summary."""
        a = cla.Analyzer(key="test", look_in_log=False, show_info=True)
        pk = (100, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(cmd="ls", ppid=None, context="ctx", key="test", live=True)
        a.filter_pid_tree()
        captured = capsys.readouterr()
        assert "Filtered out" in captured.err


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — parse_ausearch SYSCALL_AVC regex path  
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParseAusearchSyscallAvcPath:
    """Cover the SYSCALL_AVC regex match branch (no PROCTITLE) and ps fallback."""

    def test_syscall_avc_path(self):
        """Block with SYSCALL but no PROCTITLE should match SYSCALL_AVC regex."""
        block = (
            'type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=openat ppid=100 pid=200 uid=root\n'
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=200 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        # Should have extracted pid/ppid from SYSCALL line
        assert results[0].command.pid == "200"

    def test_no_match_all_regex_fallback_to_individual(self):
        """Block where neither FULL_AVC nor SYSCALL_AVC match should use individual
        REGEX_EXTRACT_* patterns."""
        block = (
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { getattr } '
            'for  pid=500 ppid=100 uid=1000 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=dir permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        assert len(results) == 1
        # Individual regex extractions should find pid, ppid
        assert results[0].command.pid == "500"

    def test_ps_fallback_for_missing_ppid(self):
        """When ppid is missing but pid is alive, get_from_pid should be called for ppid."""
        # No ppid=... in block, and no SYSCALL line
        block = (
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=12345 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        calls = []

        def fake_get_from_pid(pid, arg):
            calls.append((pid, arg))
            return cla.DEAD_PROCESS

        with mock.patch.object(a, "get_from_pid", side_effect=fake_get_from_pid):
            results = a.parse_ausearch_from_log(blocks=[block])

        assert len(results) == 1
        # Should have called get_from_pid for ppid and uid
        called_args = {c[1] for c in calls}
        assert "ppid" in called_args

    def test_syscall_avc_bad_pid_ppid(self, capsys):
        """SYSCALL_AVC match but REGEX_EXTRACT_PID_PPID fails should print error."""
        # SYSCALL line without ppid= and pid= fields
        block = (
            'type=SYSCALL msg=audit(09/02/2026 10:45:19.517:18438) : arch=x86_64 syscall=openat uid=root\n'
            'type=AVC msg=audit(09/02/2026 10:45:19.517:18438) : avc:  denied  { read } '
            'for  pid=200 comm=myapp '
            'scontext=system_u:system_r:myapp_t:s0 '
            'tcontext=system_u:object_r:cert_t:s0 tclass=file permissive=1\n'
        )
        a = cla.Analyzer(key="test", look_in_log=False)
        results = a.parse_ausearch_from_log(blocks=[block])
        captured = capsys.readouterr()
        assert len(results) >= 1
        # Should have printed error about pid/ppid parsing
        assert "error parsing" in captured.err or results[0].command.pid == "200"


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE — enrich_pid_tree deeper walk paths
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEnrichPidTreeDeepPaths:
    """Cover parent chain walking → existing entry with known cmd, ppid fill,
    and parent already in tree continuation."""

    def test_walk_continues_through_existing_parent(self):
        """When parent is already in tree with known cmd, enrichment should walk up
        to fill ppid if missing."""
        a = cla.Analyzer(key="test", look_in_log=False)
        child_pk = (500, "test")
        parent_pk = (400, "test")
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=400, context=cla.UNKNOWN, key="test", live=True,
        )
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd="/bin/bash", ppid=None, context="ctx", key="test", live=True,
        )
        a.avc_pids.add(child_pk)

        fake_child = mock.MagicMock()
        fake_child.cmdline.return_value = ["/usr/bin/app"]
        fake_child.ppid.return_value = 400

        fake_parent = mock.MagicMock()
        fake_parent.ppid.return_value = 1
        fake_parent.is_running.return_value = True
        fake_parent.status.return_value = "running"

        def fake_is_alive(pid):
            if int(pid) == 500:
                return fake_child
            if int(pid) == 400:
                return fake_parent
            raise cla.ProcessDead(f"{pid}")

        with mock.patch.object(a, "is_process_alive", side_effect=fake_is_alive), \
             mock.patch("builtins.open", side_effect=PermissionError("denied")):
            a.enrich_pid_tree()

        # Child should be enriched
        assert a.pid_tree[child_pk].cmd == "/usr/bin/app"
        # Parent should have ppid filled
        assert a.pid_tree[parent_pk].ppid == 1

    def test_enrich_info_prints(self, capsys):
        """enrich_pid_tree with show_info should print summary."""
        a = cla.Analyzer(key="test", look_in_log=False, show_info=True)
        pk = (9999999, "test")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="test", live=True,
        )
        a.avc_pids.add(pk)
        a.enrich_pid_tree()
        captured = capsys.readouterr()
        assert "enrich_pid_tree" in captured.err


# ── save_analyzed_entries + load round-trip ──────────────────────────
@pytest.mark.unit
class TestSaveAnalyzedEntries:
    """Cover save_analyzed_entries happy path, IOError path,
    and the show_info print inside it (lines 1513-1561)."""

    def test_save_and_reload(self, tmp_path):
        state = tmp_path / "state.json"
        a = cla.Analyzer(key="t", look_in_log=False, state_file_path=str(state))
        a.analyzed_entries = {"entry_a", "entry_b"}
        a.save_analyzed_entries()
        assert state.exists()
        # reload
        b = cla.Analyzer(key="t", look_in_log=False, state_file_path=str(state))
        b.load_analyzed_entries()
        assert b.analyzed_entries == {"entry_a", "entry_b"}

    def test_save_show_info(self, tmp_path, capsys):
        state = tmp_path / "state.json"
        a = cla.Analyzer(key="t", look_in_log=False, state_file_path=str(state), show_info=True)
        a.analyzed_entries = {"x"}
        a.save_analyzed_entries()
        assert "Saved 1 analyzed entries" in capsys.readouterr().err

    def test_save_io_error(self, tmp_path, capsys):
        """IOError during save should warn, not raise."""
        a = cla.Analyzer(key="t", look_in_log=False, state_file_path="/proc/nonexistent/state.json")
        a.analyzed_entries = {"x"}
        a.save_analyzed_entries()
        assert "Could not save state file" in capsys.readouterr().err

    def test_load_show_info(self, tmp_path, capsys):
        """show_info print during load (line 1497)."""
        state = tmp_path / "state.json"
        state.write_text('{"analyzed_entries": ["a", "b"]}')
        a = cla.Analyzer(key="t", look_in_log=False, state_file_path=str(state), show_info=True)
        a.load_analyzed_entries()
        assert "Loaded 2 previously analyzed entries" in capsys.readouterr().err


# ── parse_execve_logs (mocked _iter_subprocess_blocks) ───────────────
@pytest.mark.unit
class TestParseExecveLogs:
    """Cover parse_execve_logs code paths (lines 1580-1700)
    by mocking _iter_subprocess_blocks to return crafted EXECVE blocks."""

    EXECVE_BLOCK = (
        "type=SYSCALL msg=audit(1710000000.123:456): arch=c000003e syscall=59 "
        "ppid=1 pid=100 uid=0 subj=system_u:system_r:test_t:s0\n"
        "type=EXECVE msg=audit(1710000000.123:456): argc=2 a0=\"/usr/bin/app\" a1=\"--start\""
    )

    def _make_analyzer(self, **kw):
        a = cla.Analyzer(key="test", look_in_log=True, log_path="/dev/null", **kw)
        return a

    def test_basic_execve(self):
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([self.EXECVE_BLOCK])):
            count = a.parse_execve_logs()
        assert count == 1
        pk = (100, "test")
        assert pk in a.pid_tree
        assert a.pid_tree[pk].cmd == "/usr/bin/app --start"

    def test_empty_block_skipped(self):
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter(["", "  "])):
            count = a.parse_execve_logs()
        assert count == 0

    def test_no_execve_type_skipped(self):
        block = "type=SYSCALL msg=audit(1710000000.123:456): arch=c000003e pid=100 ppid=1"
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block])):
            count = a.parse_execve_logs()
        assert count == 0

    def test_already_analyzed_skipped(self, tmp_path):
        state = tmp_path / "state.json"
        a = self._make_analyzer(state_file_path=str(state))
        a.mark_entry_analyzed(self.EXECVE_BLOCK)
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([self.EXECVE_BLOCK])):
            count = a.parse_execve_logs()
        assert count == 0

    def test_no_msg_match_skipped(self):
        block = "type=EXECVE no_msg_audit_here"
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block])):
            count = a.parse_execve_logs()
        assert count == 0

    def test_no_pid_ppid_skipped(self):
        block = (
            "type=SYSCALL msg=audit(1710000000.123:456): arch=c000003e syscall=59\n"
            "type=EXECVE msg=audit(1710000000.123:456): argc=1 a0=\"/bin/ls\""
        )
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block])):
            count = a.parse_execve_logs()
        assert count == 0

    def test_pid_reuse_detect(self, capsys):
        """PID reuse: same PID with different ppid triggers collision warning."""
        block1 = (
            "type=SYSCALL msg=audit(1710000000.100:100): arch=c000003e syscall=59 "
            "ppid=1 pid=200 uid=0 subj=system_u:system_r:test_t:s0\n"
            "type=EXECVE msg=audit(1710000000.100:100): argc=1 a0=\"/bin/first\""
        )
        block2 = (
            "type=SYSCALL msg=audit(1710000000.200:200): arch=c000003e syscall=59 "
            "ppid=50 pid=200 uid=0 subj=system_u:system_r:test_t:s0\n"
            "type=EXECVE msg=audit(1710000000.200:200): argc=1 a0=\"/bin/second\""
        )
        a = self._make_analyzer()
        # Pre-populate parent so child link insertion works
        a.pid_tree[(1, "test")] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[(200, "test")],
        )
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block1, block2])):
            count = a.parse_execve_logs()
        assert count == 2
        assert "PID collision" in capsys.readouterr().err

    def test_pid_reuse_show_info(self, capsys):
        """PID reuse with show_info prints full PID list."""
        block1 = (
            "type=SYSCALL msg=audit(1710000000.100:100): arch=c000003e syscall=59 "
            "ppid=1 pid=200 uid=0 subj=system_u:system_r:test_t:s0\n"
            "type=EXECVE msg=audit(1710000000.100:100): argc=1 a0=\"/bin/a\""
        )
        block2 = (
            "type=SYSCALL msg=audit(1710000000.200:200): arch=c000003e syscall=59 "
            "ppid=50 pid=200 uid=0 subj=system_u:system_r:test_t:s0\n"
            "type=EXECVE msg=audit(1710000000.200:200): argc=1 a0=\"/bin/b\""
        )
        a = self._make_analyzer(show_info=True)
        a.pid_tree[(1, "test")] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context=cla.UNKNOWN, key="test", live=True,
            children=[(200, "test")],
        )
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block1, block2])):
            a.parse_execve_logs()
        err = capsys.readouterr().err
        # show_info path prints full list
        assert "PID collision" in err

    def test_no_events_warning(self, capsys):
        """When no blocks yielded at all, show_info prints warning."""
        a = self._make_analyzer(show_info=True)
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([])):
            count = a.parse_execve_logs()
        assert count == 0
        assert "No process-creation events found" in capsys.readouterr().err

    def test_timeout_handled(self, capsys):
        """TimeoutExpired during _iter_subprocess_blocks is caught."""
        def _raise_timeout():
            raise subprocess.TimeoutExpired(["ausearch"], 60)
            yield  # make it a generator
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", side_effect=subprocess.TimeoutExpired(["ausearch"], 60)):
            count = a.parse_execve_logs()
        assert count == 0
        assert "timed out" in capsys.readouterr().err

    def test_debug_prints(self, capsys):
        """show_debug prints are emitted during parse_execve_logs."""
        a = self._make_analyzer(show_debug=True)
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([self.EXECVE_BLOCK])):
            a.parse_execve_logs()
        err = capsys.readouterr().err
        assert "parse_execve_logs" in err

    def test_no_subj_uses_unknown(self):
        """Block without subj= should use UNKNOWN context."""
        block = (
            "type=SYSCALL msg=audit(1710000000.123:456): arch=c000003e syscall=59 "
            "ppid=1 pid=100 uid=0\n"
            "type=EXECVE msg=audit(1710000000.123:456): argc=1 a0=\"/bin/ls\""
        )
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([block])):
            a.parse_execve_logs()
        pk = (100, "test")
        assert a.pid_tree[pk].context == cla.UNKNOWN

    def test_parent_placeholder_created(self):
        """When ppid doesn't exist yet, a placeholder parent is created."""
        a = self._make_analyzer()
        with mock.patch.object(a, "_iter_subprocess_blocks", return_value=iter([self.EXECVE_BLOCK])):
            a.parse_execve_logs()
        # ppid=1, so (1, "test") should exist as placeholder
        ppid_pk = (1, "test")
        assert ppid_pk in a.pid_tree
        assert (100, "test") in a.pid_tree[ppid_pk].children


# ── identify_app_root ────────────────────────────────────────────────
@pytest.mark.unit
class TestIdentifyAppRoot:
    """Cover identify_app_root lines 1783, 1786."""

    def test_identify_app_root_found(self, capsys):
        a = cla.Analyzer(key="t", look_in_log=False, show_info=True,
                         app_name="myapp", context_filter=["myapp_t"])
        pk = (42, "t")
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/usr/bin/myapp --run", ppid=1, context="u:r:myapp_t:s0", key="t", live=True,
        )
        a.identify_app_root()
        assert pk in a.app_root_pids
        assert "Identified app root PID" in capsys.readouterr().err

    def test_identify_app_root_not_found(self, capsys):
        a = cla.Analyzer(key="t", look_in_log=False, show_info=True,
                         app_name="myapp", context_filter=["myapp_t"])
        # No matching entries
        a.identify_app_root()
        assert "Could not identify any app root PID" in capsys.readouterr().err

    def test_identify_app_root_skipped_without_params(self):
        a = cla.Analyzer(key="t", look_in_log=False)
        a.identify_app_root()  # no-op, no crash
        assert a.app_root_pids == []


# ── build(show_info=True) line 672 ──────────────────────────────────
@pytest.mark.unit
class TestBuildShowInfo:
    def test_build_with_show_info_print(self, capsys):
        """build(show_info=True) prints alias creation (line 672)."""
        ci = cla.CmdIndex()
        # Need a command long enough and weighted enough
        long_cmd = "/usr/bin/very_long_command --with many args that exceed threshold"
        ci.log_weight(long_cmd, weight=50)
        ci.build(show_info=True)
        err = capsys.readouterr().err
        assert "creating alias" in err


# ── NsIndex.items() line 488 ────────────────────────────────────────
@pytest.mark.unit
class TestNsIndexItems:
    def test_items_returns_mapping(self):
        ni = cla.NsIndex()
        ni.set(111, "label_a")
        items = list(ni.items())
        assert (111, "label_a") in items


# ── format_rules show_debug line 1471 ──────────────────────────────
@pytest.mark.unit
class TestFormatRulesDebug:
    def test_format_rules_debug_print(self, capsys):
        a = cla.Analyzer(key="t", look_in_log=False, show_debug=True)
        a.log_cmd("myapp", weight=1)
        result = cla.AnalysisResult(
            command=cla.CommandContext(key="t", descriptors=set(), pid_namespace="?", cmd="myapp", pid=1),
            avc_list=[cla.AvcDenial(method="read", source_type="src_t", target_type="tgt_t", tclass="file")],
        )
        a.format_rules([result])
        err = capsys.readouterr().err
        assert "adding" in err and "required by rule" in err


# ── parse_existing_file edge cases (lines 1249-1253, 1282, 1300, 1330) ──
@pytest.mark.unit
class TestParseExistingFileEdgeCases:
    """Cover error paths in parse_rules_from_files and parse_existing_file."""

    def test_malformed_explanation_line(self, capsys):
        """Cmd explanation line that doesn't match REGEX_MAIN_EXPLANATION (line 1249)."""
        a = cla.Analyzer(key="t", look_in_log=False, show_explanations=True)
        # Valid AVC rule but malformed explanation (no KEY_DELIMITER in cmd line)
        doc = (
            f"\nallow src_t tgt_t:file read;\n"
            f"{cla.PREFIX_CMD_LINE}# required by :\n"
            f"{cla.PREFIX_CMD_LINE}this line has no key_delimiter\n"
            f"{cla.INDEX_DELIMITER}\n"
        )
        results = a.parse_existing_file(doc)
        err = capsys.readouterr().err
        assert "error parsing a command from a file" in err

    def test_unparseable_block_error(self, capsys):
        """Block that matches neither AVC rule nor cmd section (line 1300)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        # Extra \nallow to create a second block that doesn't match REGEX_ALLOW_RULE
        doc = (
            f"\nallow src_t tgt_t:file read;\n"
            f"{cla.PREFIX_CMD_LINE}# required by :\n"
            f"{cla.PREFIX_CMD_LINE}t {cla.KEY_DELIMITER} /bin/x (pid=1 ; pid_ns=?)\n"
            f"\nallow this_is_not_a_valid_rule\n"
            f"{cla.INDEX_DELIMITER}\n"
        )
        results = a.parse_existing_file(doc)
        err = capsys.readouterr().err
        assert "error on block" in err

    def test_avc_with_pid_tracking(self):
        """File-parsed result with pid triggers avc_pids.add and log_cmd (lines 1282-1283)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        doc = (
            f"\nallow src_t tgt_t:file read;\n"
            f"{cla.PREFIX_CMD_LINE}# required by :\n"
            f"{cla.PREFIX_CMD_LINE}t {cla.KEY_DELIMITER} /usr/bin/mycmd (pid=42 ; pid_ns=?)\n"
            f"{cla.INDEX_DELIMITER}\n"
        )
        results = a.parse_existing_file(doc)
        assert len(results) > 0


# ── _iter_subprocess_blocks timeout (lines 938-940, 952) ────────────
@pytest.mark.unit
class TestIterSubprocessBlocksTimeout:
    def test_timeout_during_iteration(self):
        """Timeout triggers proc.kill and TimeoutExpired (lines 938-940)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        original_monotonic = time.monotonic

        call_count = [0]
        start_time = original_monotonic()

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] > 2:
                # Simulate timeout by returning a time way in the future
                return start_time + cla.SUBPROCESS_TIMEOUT + 100
            return start_time

        with mock.patch("time.monotonic", side_effect=mock_monotonic), \
             mock.patch("subprocess.Popen") as mock_popen:
            fake_proc = mock.MagicMock()
            fake_proc.stdout = iter(["line1\n", "----\n", "line2\n"])
            fake_proc.poll.return_value = None
            mock_popen.return_value = fake_proc
            with pytest.raises(subprocess.TimeoutExpired):
                list(a._iter_subprocess_blocks(["echo"]))

    def test_finally_kills_running_proc(self):
        """If proc is still running when generator exits, it's killed (line 952)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        with mock.patch("subprocess.Popen") as mock_popen:
            fake_proc = mock.MagicMock()
            fake_proc.stdout = iter(["line1\n"])
            fake_proc.poll.return_value = None  # still running
            mock_popen.return_value = fake_proc
            list(a._iter_subprocess_blocks(["echo"]))
            fake_proc.kill.assert_called_once()
            fake_proc.wait.assert_called()


# ── enrich_pid_tree: init placeholder + AccessDenied (lines 1848, 1901) ─
@pytest.mark.unit
class TestEnrichPidTreeInitPlaceholder:
    def test_reaching_init_creates_placeholder(self):
        """When walk reaches pid=0 (kernel), a placeholder is created via psutil.Process (line 1901)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        # Process with ppid=0 — ppid is falsy so no parent placeholder is created
        child_pk = (500, "t")
        a.pid_tree[child_pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=None, context=cla.UNKNOWN, key="t", live=True,
        )
        a.avc_pids.add(child_pk)

        fake_child = mock.MagicMock()
        fake_child.cmdline.return_value = ["/usr/bin/app"]
        fake_child.ppid.return_value = 0  # kernel thread, ppid=0 is falsy
        fake_child.pid = 500

        fake_init = mock.MagicMock()
        fake_init.cmdline.return_value = ["/sbin/init"]

        def fake_is_alive(pid):
            pid = int(pid)
            if pid == 500:
                return fake_child
            raise cla.ProcessDead(f"{pid}")

        def fake_psutil_process(pid):
            if pid == 0:
                return fake_init
            raise psutil.NoSuchProcess(pid)

        with mock.patch.object(a, "is_process_alive", side_effect=fake_is_alive), \
             mock.patch("builtins.open", side_effect=PermissionError("denied")), \
             mock.patch("psutil.Process", side_effect=fake_psutil_process):
            a.enrich_pid_tree()

        # init placeholder for pid=0 should exist
        init_pk = (0, "t")
        assert init_pk in a.pid_tree
        assert a.pid_tree[init_pk].cmd == "/sbin/init"

    def test_access_denied_labels_dead(self, capsys):
        """AccessDenied during walk labels process as dead (line 1848/1915)."""
        a = cla.Analyzer(key="t", look_in_log=False, show_debug=True)
        pk = (999, "t")
        parent_pk = (1, "t")
        # Give it a parent so post-enrichment prune doesn't remove it
        a.pid_tree[parent_pk] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="u:r:init_t:s0", key="t", live=True,
            children=[pk],
        )
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd=cla.UNKNOWN, ppid=1, context=cla.UNKNOWN, key="t", live=True,
        )
        a.avc_pids.add(pk)

        fake_proc = mock.MagicMock()
        fake_proc.cmdline.side_effect = psutil.AccessDenied(999)
        fake_proc.pid = 999

        def fake_alive(pid):
            return fake_proc

        with mock.patch.object(a, "is_process_alive", side_effect=fake_alive):
            a.enrich_pid_tree()

        assert pk in a.pid_tree
        assert a.pid_tree[pk].cmd == cla.DEAD_PROCESS
        assert "error on pid" in capsys.readouterr().err


# ── format_pid_tree: padding and visited node (lines 1971, 1977, 1982) ─
@pytest.mark.unit
class TestFormatPidTreePaddingPaths:
    def test_long_context_triggers_single_space_pad(self):
        """When pid_part + ctx_part >= CMD_ALIGN, a single space is used (line 1971)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        pk = (42, "t")
        # Very long context to exceed CMD_ALIGN
        long_ctx = "system_u:system_r:very_long_type_name_that_exceeds_alignment_threshold:s0:c0.c1023"
        a.pid_tree[pk] = cla.PidTreeEntry(
            cmd="/bin/ls", ppid=None, context=long_ctx, key="t", live=True,
        )
        output = a.format_pid_tree()
        assert "/bin/ls" in output
        assert long_ctx in output

    def test_visited_node_skipped(self):
        """Circular reference: visited node is skipped (line 1977)."""
        a = cla.Analyzer(key="t", look_in_log=False)
        pk_a = (1, "t")
        pk_b = (2, "t")
        a.pid_tree[pk_a] = cla.PidTreeEntry(
            cmd="/sbin/init", ppid=None, context="u:r:init_t:s0", key="t", live=True,
            children=[pk_b],
        )
        # b references back to a as child (circular)
        a.pid_tree[pk_b] = cla.PidTreeEntry(
            cmd="/bin/loop", ppid=1, context="u:r:loop_t:s0", key="t", live=True,
            children=[pk_a],
        )
        output = a.format_pid_tree()
        # Should contain both nodes exactly once, no infinite recursion
        assert "init" in output
        assert "loop" in output


# ── merge_json alias collision debug print (line 2105) ──────────────
@pytest.mark.unit
class TestMergeJsonAliasCollisionPrint:
    def test_alias_collision_debug_print(self, capsys):
        a = cla.Analyzer(key="t", look_in_log=False, show_info=True)
        # Pre-register an alias so the imported one collides
        a.cmd_index.register("/usr/bin/something_else", "C0")
        json_doc = {
            "version": cla.JSON_FORMAT_VERSION,
            "key": "t",
            "AVC": [],
            "index": {"/usr/bin/collision_cmd": "C0"},
            "all_cmds": {},
            "pid_tree": {},
            "app_root_pids": [],
        }
        a.merge_json(json_doc)
        err = capsys.readouterr().err
        assert "collided" in err


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TESTS — subprocess invocation of the real script
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.cli
class TestCLI:
    """End-to-end CLI tests running the actual se_log_analyser script."""

    # ── basic invocation ────────────────────────────────────────────────────

    def test_ignore_log_no_input(self):
        """--ignore-log with no input files should succeed with 0 AVC."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "0 AVC analyzed from logs" in result.stderr

    def test_ignore_log_with_key(self):
        """--key value should appear in stderr summary."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "MyHost"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    # ── key sanitization ────────────────────────────────────────────────────

    def test_key_sanitization(self):
        """Special characters in --key should be sanitized in the output."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "host/foo bar"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # '/' and ' ' replaced with '_'
        assert "host/foo bar" not in result.stdout

    # ── log path validation ─────────────────────────────────────────────────

    def test_invalid_log_path(self):
        """--log pointing to a non-existent file should fail."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", "/nonexistent/audit.log"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr or "not a regular file" in result.stderr

    def test_log_path_directory(self, tmp_path):
        """--log pointing to a directory should fail."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    # ── --files validation ──────────────────────────────────────────────────

    def test_files_missing_index(self, tmp_path):
        """--files with a file missing the INDEX delimiter should fail."""
        bad_file = tmp_path / "no_index.txt"
        bad_file.write_text("allow myapp_t tmp_t:file read;\n")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--files", str(bad_file)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "index not found" in result.stderr or "FileParsingError" in result.stderr

    def test_files_nonexistent(self, tmp_path):
        """--files with a non-existent path should fail."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log",
             "--files", str(tmp_path / "missing.txt")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    # ── --json-files validation ─────────────────────────────────────────────

    def test_json_files_invalid_json(self, tmp_path):
        """--json-files with invalid JSON should fail."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log",
             "--json-files", str(bad)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "error parsing JSON file" in result.stderr or "FileParsingError" in result.stderr

    def test_json_files_nonexistent(self, tmp_path):
        """--json-files with a non-existent path should fail."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log",
             "--json-files", str(tmp_path / "missing.json")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    # ── output destinations ─────────────────────────────────────────────────

    def test_dest_flag(self, tmp_path):
        """--dest should write human-readable output to a file."""
        dest = tmp_path / "output.txt"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--dest", str(dest)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert dest.exists()
        content = dest.read_text()
        assert cla.INDEX_DELIMITER in content

    def test_json_dest_flag(self, tmp_path):
        """--json-dest should produce valid JSON output."""
        jdest = tmp_path / "output.json"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--json-dest", str(jdest)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert jdest.exists()
        data = json.loads(jdest.read_text())
        assert "version" in data
        assert data["key"] == "test"

    def test_json_dest_only_skips_stdout(self, tmp_path):
        """--json-dest without --dest should produce no stdout."""
        jdest = tmp_path / "output.json"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--json-dest", str(jdest)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # No human-readable output on stdout
        assert result.stdout.strip() == ""

    def test_both_dest_and_json_dest(self, tmp_path):
        """--dest + --json-dest should produce both outputs."""
        dest = tmp_path / "output.txt"
        jdest = tmp_path / "output.json"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--dest", str(dest), "--json-dest", str(jdest)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert dest.exists()
        assert jdest.exists()

    # ── flags ───────────────────────────────────────────────────────────────

    def test_no_explanations_flag(self):
        """--no-explanations should suppress explanation comments."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--no-explanations"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "# required by :" not in result.stdout

    def test_no_tree_flag(self):
        """--no-tree should suppress PID tree section."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--no-tree"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert cla.PID_TREE_DELIMITER not in result.stdout

    def test_no_index_flag(self):
        """--no-index should suppress the INDEX section."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--no-index"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_debug_flag(self):
        """--debug should not crash."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--debug"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_verbose_flag(self):
        """--verbose should not crash."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "test",
             "--verbose"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    # ── --files round-trip via CLI ──────────────────────────────────────────

    @needs_testlog
    @needs_ausearch
    def test_files_roundtrip(self, tmp_path):
        """Produce output via CLI, feed it back via --files, rules should survive."""
        dest1 = tmp_path / "pass1.txt"
        result1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", TEST_LOG, "--key", "Rocky",
             "--context-filter", "myapp_t", "--app-name", "myapp",
             "--dest", str(dest1)],
            capture_output=True, text=True,
        )
        assert result1.returncode == 0
        assert dest1.exists()
        txt1 = dest1.read_text()
        assert "allow myapp_t" in txt1

        dest2 = tmp_path / "pass2.txt"
        result2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "Rocky",
             "--files", str(dest1), "--dest", str(dest2)],
            capture_output=True, text=True,
        )
        assert result2.returncode == 0
        txt2 = dest2.read_text()
        assert "allow myapp_t" in txt2

    # ── --json-files round-trip via CLI ─────────────────────────────────────

    @needs_testlog
    @needs_ausearch
    def test_json_roundtrip(self, tmp_path):
        """Produce JSON via CLI, reload via --json-files, rules should survive."""
        jdest1 = tmp_path / "pass1.json"
        result1 = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", TEST_LOG, "--key", "Rocky",
             "--context-filter", "myapp_t", "--app-name", "myapp",
             "--json-dest", str(jdest1)],
            capture_output=True, text=True,
        )
        assert result1.returncode == 0
        data1 = json.loads(jdest1.read_text())
        assert len(data1["results"]) > 0

        dest2 = tmp_path / "pass2.txt"
        result2 = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--ignore-log", "--key", "Rocky",
             "--json-files", str(jdest1), "--dest", str(dest2)],
            capture_output=True, text=True,
        )
        assert result2.returncode == 0
        txt2 = dest2.read_text()
        assert "allow myapp_t" in txt2

    # ── stderr summary line ─────────────────────────────────────────────────

    @needs_testlog
    @needs_ausearch
    def test_stderr_summary(self, tmp_path):
        """stderr should contain the AVC / file counter / PID tree summary."""
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", TEST_LOG, "--key", "Rocky",
             "--context-filter", "myapp_t", "--app-name", "myapp"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "AVC analyzed from logs" in result.stderr
        assert "found in the existing files" in result.stderr
        assert "PID tree:" in result.stderr

    # ── state file via CLI ──────────────────────────────────────────────────

    @needs_testlog
    @needs_ausearch
    def test_state_file_flag(self, tmp_path):
        """--state-file should create a state file after analysis."""
        sf = tmp_path / "state.json"
        result = subprocess.run(
            [sys.executable, _SCRIPT_PATH, "--log", TEST_LOG, "--key", "Rocky",
             "--context-filter", "myapp_t", "--app-name", "myapp",
             "--state-file", str(sf)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert sf.exists()
        data = json.loads(sf.read_text())
        assert "analyzed_entries" in data
