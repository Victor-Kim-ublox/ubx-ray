# admin_cli.py

Server-side CLI for inspecting and deleting stored upload data. Runs directly
against `data/ubxray.sqlite3`, `uploads/`, and `outputs/` -- no HTTP layer, no
admin token needed. Meant to be invoked from the server shell.

Safety model: every destructive subcommand is **dry-run by default**. Re-run
with `--yes` to actually delete. `purge --all` additionally requires
`--i-mean-it`.

## Subcommands

### `list`
List results in the DB, newest first. Shows on-disk footprint per row and a
grand total.

```
python admin_cli.py list
python admin_cli.py list --user <user_id>
python admin_cli.py list --status error --limit 20
```

### `show <rid>`
Dump the DB row plus every file under `uploads/` and `outputs/` that matches
the rid. Useful before deleting.

### `delete <rid> [<rid> ...]`
Delete the given rids. Rows with `status` in `queued` / `running` are skipped
unless `--force`. Works on orphans too (files on disk with no DB row).

```
python admin_cli.py delete aa11 bb22
python admin_cli.py delete aa11 bb22 --yes
python admin_cli.py delete aa11 --yes --force
```

### `purge` (bulk delete)
Filter-based deletion. Requires at least one of `--user`, `--older-than`,
`--all`.

| Flag | Effect |
|---|---|
| `--user <uid>` | only rows belonging to this user |
| `--older-than <N>` | only rows with `uploaded_at` more than N days old |
| `--all` | every row -- also requires `--i-mean-it` |
| `--include-running` | don't skip queued/running rows (default: skip) |
| `--yes` | actually delete (default: dry-run) |

```
python admin_cli.py purge --older-than 7 --yes
python admin_cli.py purge --user 3fe08a... --yes
python admin_cli.py purge --all --yes --i-mean-it
```

### `prune-orphans`
Remove files in `uploads/` and directories under `outputs/` that don't have a
matching DB row. Useful after a crash or manual DB edit.

```
python admin_cli.py prune-orphans
python admin_cli.py prune-orphans --yes
```

## Notes

- The DB `results` row is removed in the same transaction as the on-disk
  files, mirroring the web app's `cln_delete_by_rid` helper.
- Concurrent runs with the live web server are safe (SQLite WAL), but a row
  being actively written to by an in-flight upload could race -- prefer
  `delete --force` only when you know the write is stuck.
- No configuration file: the script resolves paths relative to its own
  location (`data/`, `uploads/`, `outputs/` next to `admin_cli.py`).
