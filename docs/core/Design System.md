# F-1 Waypoint: Design System

The product is read by anxious people checking a rule that could affect their status. The design has to feel calm, precise and honest about what it does not know. Version 1 is light mode only.

## Principles

1. **Source first.** The citation is the hero, not decoration. Every answer shows where it came from.
2. **The answer belongs to the question.** The question is the largest text on the screen, and the answer visibly responds to it.
3. **Calm under stakes.** No alarm colors for normal content. Red is for real errors only.
4. **Refusal is a feature.** A refusal is a designed, respectful state, not an error page.
5. **Not official.** Nothing suggests government affiliation: no seals, flags, eagles or .gov-style banners. The product says plainly that it is an independent tool.
6. **Scannable deadlines.** Numbers and windows get their own visual treatment so a worried reader finds them in two seconds.
7. **Mobile first.** Most students arrive on a phone, often from a search or a forum.
8. **No generic AI look.** No chat bubbles, sparkle icons, gradient text or glow, and no fintech-dashboard panels.

## Color

One accent, near-neutral surfaces, and state colors only for state. Text pairs pass WCAG AA, and deadline, refusal, caution and source tier each carry an icon or a text label, never color alone.

| Token | Value | Use |
|---|---|---|
| Ground | `#F2F5F5` | Page background, almost neutral with a faint teal tint |
| Card | `#FFFFFF` | Cards and the input |
| Track | `#E4E9EA` | Tab track, caution box |
| Text / muted | `#11171B` / `#365258` | Body and secondary text |
| Accent | `#005E6E` | Buttons, links, active tab, refusal outline |
| Accent tint | `#CEEFF6` | Deadline row, highlighted sentence |
| Success | `#DCF7E1` / `#0D4A22` | Verified badge |
| Failure | `#FFEDEB` / `#9E1618` | True errors only |

Caution has no color of its own: a neutral box with a ▲ icon. There is no amber anywhere.

## Typography and spacing

- System UI fonts for text, so first paint is fast on weak connections. The wordmark alone is set in a serif. Monospace is used inline for form numbers and citations, so "I-983" reads as an exact term.
- Four sizes (14, 16, 20, 25) and two weights (400, 600). Nothing below 14px.
- Spacing scale: 4, 8, 12, 16, 24, 32. Related items sit 8px apart, groups 24px apart. Cards have a 16px radius and controls 12px.
- Separation comes from tint and a soft shadow, not borders. The deliberate exceptions are the input outline and the refusal card outline.

## Screens

- **Home.** Header with the wordmark and a verified badge. Two tabs: Ask and Application form help. Then the input with the Ask button as a separate full-width button beneath it, a one-line privacy hint, a short "Try asking" list, a coverage note, and a text link to the employer brief.
- **Answer.** The question appears as a large heading above the card, joined to it by a dot and a thin line, and the phrase it shares with the answer is underlined in both. The card leads with a key fact row (the figure, the rule, who to report to), then an answer sentence that reuses the question's own words, then the sources.
- **Sources.** One tappable row per source. Each states its tier in plain words, "Official source" or "University guide, not official", with a distinct marker. Tapping opens the exact passage with the supporting sentence highlighted, the official link and the verified-on date.
- **Refusal.** A white card with a teal outline and a teal title, under the same question heading. Three plain groups: what the sources say, what can't be decided here, and who can decide.
- **Verified badge.** Shows the verified-on date, and turns into a neutral "Sources may be out of date" notice after 14 days.
- **Application form help and employer brief.** Cited, copyable scripts for the two application questions, and a one-page on-screen brief.

## Voice

- Plain English, short sentences, terms defined on first use (DSO, SEVP, EAD, I-983).
- Lead with the answer, then the rule, then the source.
- State numbers exactly as the source does. Never round or soften a deadline.
- No reassurance the sources cannot back.
- Refusals are specific and kind: say what the sources do say, say what cannot be decided here and why, say who can decide.
- No em dashes in UI copy or docs.

## Accessibility and responsiveness

- WCAG 2.2 AA target, visible focus rings, touch targets of at least 44 by 44 pixels, landmarks, and an answer region announced politely to screen readers.
- Every control works by keyboard. Motion is short, ease-out and functional, and off under reduced-motion.
- Under 640px: single column, and the source sheet becomes a bottom sheet. Above 1024px: the answer column plus a persistent source panel.

## Brand

Name: F-1 Waypoint ("Waypoint" in tight spaces). A simple serif wordmark in the accent color. No shields, stars or seals. Footer: "Independent project. Not affiliated with USCIS, DHS, or any university."
