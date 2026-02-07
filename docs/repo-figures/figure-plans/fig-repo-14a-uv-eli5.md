# fig-repo-14a: Reproducibility: The Dice Game (ELI5)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-14a |
| **Title** | Reproducibility: The Dice Game |
| **Complexity Level** | L0 (ELI5 - Concept only) |
| **Target Persona** | PI, First-year intern, Non-technical |
| **Location** | Root README, docs/getting-started/ |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the reproducibility problem with package managers using a simple dice analogy—NO code, NO technical details.

## Key Message

"pip is like rolling dice—you might get different results each time. uv is like a locked recipe—same ingredients, every time."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REPRODUCIBILITY: THE DICE GAME                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                                                                                 │
│    ❌ OLD WAY (pip)                      ✅ NEW WAY (uv)                        │
│                                                                                 │
│    🎲 🎲 🎲                               🔒                                    │
│    Rolling dice                          Locked recipe                          │
│                                                                                 │
│    "Install packages"                    "Install packages"                     │
│         ↓                                     ↓                                 │
│    Machine A: 📦📦📦                     Machine A: 📦📦📦                      │
│    Machine B: 📦📦📦📦                   Machine B: 📦📦📦                      │
│    Machine C: 📦📦                       Machine C: 📦📦📦                      │
│                                                                                 │
│    DIFFERENT!                            IDENTICAL!                             │
│    "It works on my machine..."           "Works everywhere!"                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    SPEED BONUS                                                                  │
│                                                                                 │
│    pip:  🐢░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  60 seconds                          │
│    uv:   🚀██░░░░░░░░░░░░░░░░░░░░░░░░░░░   5 seconds                          │
│                                                                                 │
│                    10× FASTER!                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements (MAX 5 CONCEPTS)

1. **Dice vs Lock metaphor**: Random vs deterministic
2. **Three machines comparison**: Different results vs identical
3. **"Works on my machine" problem**: Named and illustrated
4. **Speed comparison**: Turtle vs rocket (visual)
5. **10× faster benefit**: Single number to remember

## Text Content

### Title Text
"Reproducibility: The Dice Game"

### Labels/Annotations
- NO code snippets
- NO technical terms (sub-dependencies, lockfile, etc.)
- Simple icons: dice, lock, turtle, rocket, package boxes

### Caption
When you install packages the old way (pip), different computers might get different versions—like rolling dice. With uv, everyone gets exactly the same packages—like following a locked recipe. Bonus: it's 10× faster!

## Prompts for Nano Banana Pro

### Style Prompt
Simple, friendly infographic for non-technical audience. Minimal design with large icons. Use playful but professional colors—muted greens and reds for good/bad. Dice and lock as central metaphors. NO code, NO technical jargon. Speed comparison with turtle and rocket icons. Clean, uncluttered layout. Medical research context but approachable.

### Content Prompt
Create a simple two-column comparison:

**LEFT (pip - red/orange tint)**:
- Large dice icons (3 dice)
- Three machine icons with DIFFERENT numbers of package boxes
- Label: "Different results each time"
- Quote: "It works on my machine..."
- Turtle icon for speed

**RIGHT (uv - green/blue tint)**:
- Large lock icon
- Three machine icons with IDENTICAL package boxes
- Label: "Same results everywhere"
- Quote: "Works everywhere!"
- Rocket icon for speed

**BOTTOM**:
- Speed bar: turtle (long bar, 60s) vs rocket (short bar, 5s)
- Big text: "10× FASTER!"

NO code, NO technical terms, NO jargon.

## Alt Text

Simple comparison showing pip (represented by dice, random results across three machines) versus uv (represented by a lock, identical results across three machines). Speed comparison shows pip as turtle (60 seconds) and uv as rocket (5 seconds, 10× faster).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README
