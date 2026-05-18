---
name: conventional-commit
description: Use this skill any time the user asks for a git commit, to commit changes, or to write a commit message. It writes the message in Conventional Commit format.
---

When you create a git commit, follow these rules. Never add a "Co-Authored-By" trailer.

1. Always write in English — subject line and body.
2. Start the subject line with one of: feat, fix, chore, docs, refactor, test, perf.
3. Add a colon and a space, then a short imperative summary, no period.
4. Keep the subject under 70 characters.
5. If the change touches more than two files, add a one-line body that says why.

Example:

  feat: add IndexNow ping to publish workflow

  Auto-pings Bing on every push to main so new posts get indexed faster.
