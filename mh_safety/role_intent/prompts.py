"""Generation system prompt for the role x intent study.

A generic assistant -- the realistic default deployment context. Only the user
message (role x intent framing) varies. Set NEUTRAL_SYSTEM = "" for a fully bare
run, or swap in a safety-primed / therapist prompt to compare.
"""

NEUTRAL_SYSTEM = "You are a helpful assistant. Reply to the user's message."
