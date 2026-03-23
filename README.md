# SELinux Portability and Explainability Tools

A set of 3 tools aiming to assist SELinux policy creation in complement of existing ones. It focuses on two gaps: **explainability** and **portability**. The first tool is in charge of gathering contexts for technical support and informed sorting of audit2allow rules. The others convert the output of this file, merge them with existing policies and make them loadable on different hosts for portability.

## Gaps Addressed in Classical SELinux Tools

**Explainability Gap:**
Classical tools like `audit2allow` generate policy rules efficiently but provide no custom context about *why* each rule is needed. When reviewing hundreds of generated rules, administrators lack information about which processes triggered them, what commands were running, or how they relate to each other. This makes it difficult to:
- Identify which rules are legitimate vs. potential security issues
- Provide meaningful explanations to security teams or auditors
- Debug policy problems or understand application behavior

**Portability Gap:**
SELinux policies often reference types that exist on one system but not another (due to different installed packages, OS versions, or custom policies). When deploying policies across environments, missing type definitions cause loading failures. Classical tools lack mechanisms to:
- Automatically makes the policy and SELinux environment compatible for portability
- Clean policies of environment-specific references

## Overview

These tools work together to provide a complete workflow for creating, analyzing, and deploying SELinux policies across different systems:

1. **`se_log_analyser`** - Enhances `audit2allow` output with process context and command hierarchies
2. **`se_policy_merger`** - Consolidates multiple policy sources into unified `.te` files
3. **`se_check_type`** - Ensures type compatibility across target systems

## Tools

### 1. se_log_analyser

Analyzes SELinux audit logs (from `/var/log/audit/audit.log`) and generates policy recommendations enriched with process context, command information, and hierarchical relationships. Supports incremental analysis, file merging, and both human-readable and JSON output formats.

**Requirements:**
- Python >= 3.9
- psutil (Python module)
- aureport
- ausearch
- audit2allow
- se_log_analyser use auditd execve logs (`auditctl -a always,exit -S execve`)

**Usage Examples:**

```bash
# Basic analysis
se_log_analyser --key $hostname --log ./logs --dest rules_with_explication.txt

# Incremental analysis (remembers what was already analyzed)
se_log_analyser --key $hostname --log ./logs --state-file ./analyzer_state.json --dest rules.txt

# Without PID tree
se_log_analyser --key $hostname --log ./logs --files ./file1 ./file2 --no-tree --dest rules.txt

# JSON output (structured, for programmatic use)
se_log_analyser --key $hostname --log ./logs --json-dest ./output.json

# Mixed: human-readable files + JSON files as input, both outputs
se_log_analyser --key $hostname --files rules.txt --json-files prev.json --dest merged.txt --json-dest merged.json
```

**Limitations:**
- Currently only understands "allow rules" (some AVC/USER_AVC errors may not be resolved)
- Does not process SELINUX_ERR or USER_SELINUX_ERR log messages
- Use standard `audit2allow` for advanced cases like invalid contexts or constraint violations

### 2. se_policy_merger

Consolidates multiple SELinux policy files written in human-readable `.te` format into a unified policy. Deduplicates rules, generates proper `require` blocks, and can directly process `se_log_analyser` output.

**Supported Rule Types:**
- `allow` rules
- `role` rules
- `type_transition` rules
- `role_transition` rules
- `typeattribute` statements
- `type` definitions

**Usage Examples:**

```bash
# Merge multiple policy files
se_policy_merger --files ./file1 ./file2 -v > rules.te
```

### 3. se_check_type

Validates that all types referenced in a SELinux policy exist on the target host. Can automatically create missing types or remove invalid references, enabling cross-system policy deployment.

**Usage Examples:**

```bash
# Check policy and show missing types
se_check_type --policy myapp.te

# Remove missing types from the policy
se_check_type --policy myapp.te -r

# Create and load missing types
se_check_type --policy myapp.te -c

# Debug mode
se_check_type --policy myapp.te -d

# Ignore specific types
se_check_type --policy myapp.te --ignore type1 type2
```

**Note:** Use the `--ignore` option with caution as it can lead to policy loading errors.

## Typical Workflow

1. **Analyze audit logs** on the source system:
   ```bash
   se_log_analyser --key production-server --state-file state.json > analysis.txt
   ```

2. **Merge policies** if you have multiple sources or existing policies:
   ```bash
   se_policy_merger --files analysis.txt existing_policy.te > merged.te
   ```

3. **Validate and fix types** for the target system:
   ```bash
   se_check_type --policy merged.te -c
   ```

4. **Load the policy** on the target system using standard SELinux tools

## Author

Basile Leretaille

## License

Copyright (c) 2026 Hewlett Packard Enterprise Development LP

Licensed under the MIT License. See individual script headers for full license text.
