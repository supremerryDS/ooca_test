"""Allow `python -m triage_agent`, same as the `triage-agent` command."""

import sys

from triage_agent.cli import main

sys.exit(main())
