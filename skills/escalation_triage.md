# Escalation Triage Skill

## Objective
Review customer operations requests and decide which ones need escalation.

## Procedure
1. Read every record you are given. Use only these records.
2. Skip records whose resolution_status is `resolved`.
3. Give each remaining record an escalation level:
   - **ESCALATE NOW**: priority is `critical`, OR priority is `high` and
     resolution_status is `open`.
   - **WATCH**: priority is `high` and resolution_status is `in_progress`,
     OR contract_value is 200000 or more.
   - **NORMAL**: everything else.
4. Mark a record as **VIP** when contract_value is 200000 or more.
5. For each ESCALATE NOW record, suggest one next action for its assigned_team.

## Output format
1. A table with columns: ID | Customer | Company | Priority | Status | Level | VIP
   Sort it ESCALATE NOW first, then WATCH, then NORMAL.
2. "Next actions": one bullet per ESCALATE NOW record.
3. One-line summary, e.g. "3 to escalate, 2 to watch, 1 normal."

## Rules
- Never invent records or field values.
- If no records were given, say so and stop.
- Do not claim you changed anything in the database. You only recommend.
