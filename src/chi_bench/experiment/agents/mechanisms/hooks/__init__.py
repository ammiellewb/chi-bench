"""Standalone Claude Code hook scripts uploaded into the agent container.

Every module here must stay **stdlib-only** and runnable both as a package
module (for unit tests) and as a bare script via ``python3 -I <path>`` (how
Claude Code invokes it inside ``chi-bench:agent``). Shared helpers live in
``mech_common`` and are imported through the dual-mode shim at the top of
each script.
"""
