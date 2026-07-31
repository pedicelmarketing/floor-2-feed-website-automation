<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Explain in plain language

Default to simple, non-technical explanations in anything user-facing: chat replies,
summaries, published pages, PR descriptions.

- Lead with what a thing *does* and why it matters, not what it is called.
- Spell out a term the first time it appears, or avoid it. No unexplained jargon
  (denoise, VACE, Spearman, ControlNet, blockout) — if it earns its place, define it in
  the same sentence.
- Short sentences. Concrete numbers over adjectives.
- Analogies are welcome when they are accurate.

This is about *wording*, not rigour. Keep stating what was actually measured versus
assumed, what failed, and what has not been verified — just say it plainly. Never simplify
by dropping a caveat or overstating a result.

Code comments and commit messages stay technical; the audience there is different.

# Always show your work as a Claude artifact

Any result worth reporting gets published as an artifact, not described in chat and not left
as a file path. Videos, renders, comparisons, measurements, research — publish it and give the
link.

- **Never hand over a local file path.** It cannot be opened from the other side of the
  conversation. Publishing is the only way the work is actually seen.
- Embed the media in the page. A comparison of four videos means four videos playing on one
  page, not four numbers and a description of what they look like.
- Republish the same file path to update in place, so a link that was shared stays current
  rather than going stale while a newer link exists elsewhere.
- The artifact carries the caveats too — what was assumed, what is uncalibrated, what a
  number does not establish. Showing the work means showing where it is weak.
