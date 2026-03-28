# Writing Tone and Style Guide

This document defines the writing standards for all dissertation chapters and research documents in this project. Every section written must follow these principles without exception.

---

## 1. One or Two Ideas Per Sentence

Each sentence carries one idea, or at most two closely related ideas joined by a simple connective such as "and" or "but". If a sentence is trying to say three things, it should be three sentences. Long, nested sentences hide reasoning and exhaust the reader.

**Do not write:**
> The stochastic nature of overtake resolution, combined with the non-stationarity introduced by concurrent opponent adaptation, means that Q-value estimates formed under one opponent policy become unreliable as that policy evolves, creating a compounding instability that uniform replay buffers are not designed to handle.

**Write instead:**
> Overtake outcomes are stochastic. When the opponent is also learning, the environment that any one agent experiences keeps changing. Q-value estimates formed under an older opponent policy can become unreliable quite quickly. Uniform replay buffers are not designed to handle this kind of instability.

---

## 2. Plain Language Over Buzzwords

Use the simplest word that is still accurate. Avoid academic jargon where plain English communicates the same idea. Technical terms that have a precise meaning in the field are permitted and should be used correctly, but they must be explained when first introduced.

**Avoid:** leverage, operationalise, paradigm, synergise, holistic, robust (as a vague filler), novel (as a self-descriptor), cutting-edge.

**Use instead:** use, apply, framework, combine, complete, strong, new, recent.

When a technical term appears for the first time, define it in the same sentence or the one immediately following.

---

## 3. No Semicolons, Colons, Em-Dashes, or Hyphens in Prose

Prose sentences must not contain semicolons, colons, em-dashes, or hyphens.

If a semicolon is joining two related clauses, use a full stop and start a new sentence instead.

If a colon is introducing a list or explanation, restructure the sentence so that the list follows naturally, or use a paragraph break.

If an em-dash is used as an aside or parenthetical, use a new sentence or a pair of parentheses instead.

Hyphens in established technical compound terms are permitted where the compound is a recognised term in the field (for example, "Q-value" or "multi-agent"). Hyphens in descriptive phrases should be avoided by rephrasing (for example, "a zone with high value" rather than "a high-value zone").

---

## 4. British English Throughout

Use British spelling at all times.

Key examples: behaviour (not behavior), colour (not color), analyse (not analyze), organise (not organize), programme (not program), modelling (not modeling), prioritised (not prioritized), neighbourhood (not neighborhood), centre (not center).

Use single quotation marks for quotations within the text. Use the Oxford comma in lists of three or more items.

---

## 5. Accessible, Journey-Style Writing

The reader is a capable person who is new to this topic. The writing should take them on a clear intellectual journey. Each section should answer the question "why does this matter?" before explaining "what is it?" and "how does it work?".

Avoid walls of text. Aim for paragraphs of three to five sentences. Vary sentence length. Short sentences after a longer explanation create emphasis and allow the reader to breathe.

Introduce context before detail. Do not assume the reader knows why something is interesting. Show them first.

---

## 6. Senior Machine Learning Researcher Voice

Write with the confidence and authority of an experienced researcher, but without arrogance or unnecessary complexity. Make claims precisely. Acknowledge limitations honestly.

Do not hedge every sentence. If the evidence supports a claim, state it clearly. If the evidence is limited, say so once and move on. Repeated hedging ("it may perhaps be possible that") undermines the reader's trust in the analysis.

Do not use the first person in the main text unless reporting specific design decisions. Use the passive or impersonal third person for describing methods and results.

---

## 7. Every Claim Must Be Cited

Every factual claim, statistical assertion, or reference to prior work must be backed by a citation from a published paper or book. Uncited claims are not permitted.

This project follows Leeds Harvard referencing throughout. The format guide is available at: https://library.leeds.ac.uk/referencing-examples/9/leeds-harvard

**In-text citation format:**
- Parenthetical: (Author, Year) or (Author and Author, Year) or (Author et al., Year)
- Narrative: Author (Year) showed that...
- In LaTeX: use \citep{} for parenthetical and \citet{} for narrative

**If a claim lacks an existing citation:**
Find a published paper that either supports or refutes the claim. A paper that refutes or complicates a claim is equally valid and often more informative. The goal is that every statement the reader might question has a pointer to further reading. Papers from any peer-reviewed venue are acceptable, including conference proceedings, journals, and well-established preprints.

---

## 8. Truth and Evidence-Based Statements Only

Do not assert anything that cannot be verified from the data or from the cited literature. Do not speculate without flagging it explicitly as a hypothesis or prediction.

When describing experimental results, report what was observed and what the data show. Interpretation is permitted but must be distinguished from observation. Use phrases such as "this is consistent with", "one explanation is", or "the data suggest" when moving from observation to interpretation.

Do not overstate findings. If a result holds across three seeds under two stochasticity levels, report that. Do not claim it generalises to all conditions unless the evidence supports that.

---

## 9. Do not start sentences with 'And'

## Quick Reference Checklist

Before submitting any section, verify the following:

- [ ] No sentence contains more than two ideas
- [ ] All jargon is defined on first use
- [ ] No semicolons, colons, em-dashes, or prose hyphens
- [ ] British spelling throughout
- [ ] Every paragraph has a clear purpose and advances the reader's understanding
- [ ] Every factual claim has a citation
- [ ] No uncited speculation presented as fact
- [ ] Results are reported accurately without overstatement
