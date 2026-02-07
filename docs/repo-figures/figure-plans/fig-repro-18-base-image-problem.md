# fig-repro-18: The Base Image Problem

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-18 |
| **Title** | The Base Image Problem |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, DevOps |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain how Docker base image tags (e.g., python:3.11) resolve to different images over time, and how to pin to specific digests.

## Key Message

"'FROM python:3.11' isn't a version—it's a pointer that changes weekly. Pin to digest (sha256:abc...) or your Dockerfile becomes non-reproducible."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE BASE IMAGE PROBLEM                                       │
│                    Tags drift, digests don't                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT HAPPENS WITH FLOATING TAGS                                                │
│  ════════════════════════════════                                               │
│                                                                                 │
│  Your Dockerfile:                                                               │
│  ┌────────────────────────────────────────┐                                    │
│  │  FROM python:3.11                       │                                    │
│  │  RUN pip install pandas                 │                                    │
│  │  COPY . /app                            │                                    │
│  └────────────────────────────────────────┘                                    │
│                                                                                 │
│  What "python:3.11" resolves to:                                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  January 2024:   python:3.11.7 + Debian Bookworm + OpenSSL 3.0.10      │   │
│  │        │                                                                │   │
│  │        ▼                                                                │   │
│  │  April 2024:     python:3.11.8 + Debian Bookworm + OpenSSL 3.0.11      │   │
│  │        │                                                                │   │
│  │        ▼                                                                │   │
│  │  October 2024:   python:3.11.9 + Debian Bookworm + OpenSSL 3.0.13      │   │
│  │        │                                                                │   │
│  │        ▼                                                                │   │
│  │  January 2025:   python:3.11.10 + Debian ??? + OpenSSL ???             │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Same tag, FOUR different images!                                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON TAGS AND THEIR DANGERS                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  │ Tag                    │ What it means           │ Danger level  │          │
│  │ ─────────────────────── │ ─────────────────────── │ ───────────── │          │
│  │ python:latest          │ Whatever is newest      │ 🔴 EXTREME    │          │
│  │ python:3               │ Any 3.x (3.11? 3.12?)   │ 🔴 HIGH       │          │
│  │ python:3.11            │ Any 3.11.x              │ 🟡 MEDIUM     │          │
│  │ python:3.11.7          │ Specific patch          │ 🟢 LOW        │          │
│  │ python:3.11.7@sha256:  │ Exact image digest      │ ✅ NONE       │          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE SOLUTION: PIN TO DIGEST                                                    │
│  ════════════════════════════                                                   │
│                                                                                 │
│  Step 1: Find the current digest                                                │
│  $ docker pull python:3.11.7-slim-bookworm                                      │
│  $ docker inspect python:3.11.7-slim-bookworm --format='{{.RepoDigests}}'       │
│  → sha256:abc123def456...                                                       │
│                                                                                 │
│  Step 2: Pin in Dockerfile                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  # DON'T:                                                               │   │
│  │  FROM python:3.11                                                       │   │
│  │                                                                         │   │
│  │  # DO:                                                                  │   │
│  │  FROM python:3.11.7-slim-bookworm@sha256:abc123def456...                │   │
│  │  #    └─── readable tag ───────┘  └─── immutable digest ──┘             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  The digest is a cryptographic hash of the image contents                       │
│  → Same digest = EXACTLY the same image, forever                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR PRACTICE                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  1. Pin base images to digest in Dockerfile                                     │
│  2. Document the digest + date in docs/environment.md                           │
│  3. Review and update digests quarterly (security patches)                      │
│  4. When updating, rebuild and retest before committing                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Timeline diagram**: Same tag resolving to different images over time
2. **Danger level table**: Common tags ranked by risk
3. **Step-by-step solution**: How to get and pin digest
4. **Dockerfile example**: Before/after with annotations

## Text Content

### Title Text
"The Base Image Problem: Tags Drift, Digests Don't"

### Caption
Docker tags like 'python:3.11' are mutable pointers that resolve to different images over time. January's python:3.11 differs from October's. Pin to immutable digests (sha256:abc...) to freeze the exact image. Foundation PLR pins all base images to digests and documents update procedures.

## Prompts for Nano Banana Pro

### Style Prompt
Timeline showing tag resolution changing over time. Danger level table with color-coded risk. Terminal commands for getting digest. Dockerfile before/after comparison. Technical, clean style.

### Content Prompt
Create "Base Image Problem" infographic:

**TOP - Timeline**:
- "FROM python:3.11" at top
- Arrow pointing to 4 different images over time (Jan-Oct)
- "Same tag, 4 different images!" callout

**MIDDLE - Danger Table**:
- 5 rows from :latest (extreme danger) to @sha256 (none)
- Color-coded risk levels

**BOTTOM - Solution**:
- Terminal commands to get digest
- Dockerfile before (tag only) vs after (tag + @sha256)

## Alt Text

Base image problem infographic. Dockerfile with FROM python:3.11 resolves to different images over time: 3.11.7 in January, 3.11.8 in April, 3.11.9 in October, 3.11.10 in next January. Danger level table: python:latest (extreme), python:3 (high), python:3.11 (medium), python:3.11.7 (low), python:3.11.7@sha256 (none). Solution: use docker inspect to get digest, then FROM python:3.11.7-slim-bookworm@sha256:abc123. Digest is cryptographic hash ensuring exact same image forever.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

