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
