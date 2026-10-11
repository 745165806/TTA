# Branch and worktree management

`main` is the stable baseline for new EP-TTA research. Start each new study from
the latest `origin/main` in its own branch and worktree:

```bash
git fetch origin main
git worktree add -b codex/ep-tta-next-YYYYMMDD \
  .worktrees/ep-tta-next-YYYYMMDD origin/main
```

The primary worktree is now clean on `main`; keep each new study in its own
worktree so the local ignored research artifacts remain untouched. Record the
exact source commit in each run's metadata. Research methods remain in
`experiments/` until their protocol, tests and access boundaries justify
promotion to a default method.

## Historical branches

[The research archive index](../RESEARCH_ARCHIVE_INDEX.md) records each
historical branch tip and annotated tag. A tag preserves committed Git history
only. Ignored checkpoints, scores, caches, private files and uncommitted work
need separate protection. Before retiring one worktree, verify:

1. Its current HEAD equals the original commit recorded by a remotely
   recoverable annotated tag. Keep private Meta Audio history on its restricted
   local recovery path until a suitable private remote or independent recovery
   plan is verified.
2. Tracked status, untracked paths and ignored paths have been inventoried.
   The protected source and backup file sets, types, sizes and SHA-256 values
   agree. Important non-reproducible results have a verified second copy on an
   independent physical device or trusted remote, plus a restore check.
3. No experiment process, CUDA job or active Codex task uses the worktree.
4. Required reusable code and scientific conclusions are in `main` or clearly
   accessible through the archive tag; historical negative results remain
   identifiable.

Only then remove that one clean worktree with `git worktree remove`, delete its
local historical branch, recheck the remote tag and delete any corresponding
remote experiment branch. Never use forced worktree removal, broad cleanup
commands or branch-name patterns as evidence of safety. The primary worktree
switched to `main` only after its 20 former dirty paths were independently
backed up and restored, and its large ignored results were checked in place.
Private Meta Audio Git objects stay in restricted local bundles and recovery
mirrors, never on the public remote.

The local artifact inventory and exact backup paths are restricted operational
records and are not published in this repository.
