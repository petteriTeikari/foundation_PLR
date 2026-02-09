# fig-nb-06b: Quarto Security -- What Changed, What Didn't

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-06b |
| **Title** | Quarto Security: What Changed, What Didn't |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | ML Engineer, Security Engineer |
| **Location** | notebooks/README.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Companion to fig-nb-06 (which covers Jupyter/notebook attack surface broadly). This figure specifically addresses Quarto's security posture: what Jupyter vulnerabilities it eliminates, what persists because code must still execute, and what NEW attack surface Quarto introduces through its Lua filters, extension ecosystem, and dependency chain. Readers should understand that moving to Quarto is not a security silver bullet.

## Key Message

"Quarto eliminates Jupyter's stored-output XSS and server-side attack surface, but introduces new vectors through unsandboxed Lua filters, a weak extension trust model, and a deeper dependency chain (Pandoc, Deno, LuaTeX). The only real-world Quarto CVE so far: a polyfill.io supply chain compromise affecting every HTML document."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| CVE-2024-38526 | Polyfill.io CDN compromise via Quarto's MathJax template; affected all HTML output | [Posit Support](https://support.posit.co/hc/en-us/articles/24767859071895-PolyFill-CDN-Supply-Chain-Attack-CVE-2024-38526) |
| CVE-2023-35936/38745 | Pandoc arbitrary file write via crafted images in PDF/extract-media | [GitHub Advisory GHSA-xj5q-fv23-575g](https://github.com/jgm/pandoc/security/advisories/GHSA-xj5q-fv23-575g) |
| CVE-2023-32700 | LuaTeX shell command execution even with --no-shell-escape | [TUG Security](https://tug.org/~mseven/luatex.html) |
| CVE-2025-51591 | Pandoc SSRF via iframe src fetching internal URLs (AWS IMDS theft) | [The Hacker News](https://thehackernews.com/2025/09/hackers-exploit-pandoc-cve-2025-51591.html) |
| Deno CVEs | Multiple sandbox bypasses: FFI (CVE-2022-24783), symlink (CVE-2021-41641), proc (CVE-2024-34346), ANSI (CVE-2024-32477) | [Deno Security](https://github.com/denoland/deno/security/advisories) |
| quarto-cli#8274 | LUA_CPATH confusion: Pandoc's Lua accidentally loads libraries from system path | [GitHub Issue](https://github.com/quarto-dev/quarto-cli/issues/8274) |
| Pulse Security | Go-pandoc RCE through Lua filter injection | [Advisory](https://pulsesecurity.co.nz/advisories/gopandoc-filter-rce) |
| quarto-cli security | Zero published security advisories; no SECURITY.md file | [GitHub Security Tab](https://github.com/quarto-dev/quarto-cli/security) |
| Pandoc Manual | --sandbox does NOT restrict Lua filters or PDF production | [pandoc.org/MANUAL.html](https://pandoc.org/MANUAL.html) |
| Quarto Extensions | Trust model: single Y/n prompt, no code signing, no permissions | [Managing Extensions](https://quarto.org/docs/extensions/managing.html) |
| quarto-live | WebAssembly-based execution via webR/Pyodide (browser sandbox) | [quarto-live docs](https://r-wasm.github.io/quarto-live/) |
| Jiang et al. 2025 | Jupyter CVE count doubled between 2023-2024 | [arXiv:2507.18833](https://arxiv.org/abs/2507.18833) |

## Visual Concept

```
+---------------------------------------------------------------------------------+
|               QUARTO SECURITY: WHAT CHANGED, WHAT DIDN'T                        |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  THREE-COLUMN LAYOUT: FIXED | PERSISTS | NEW                                   |
|  ─────────────────────────────────────────                                      |
|                                                                                 |
|  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 |
|  │  ✓ FIXED         │  │  ⚠ PERSISTS     │  │  ✗ NEW RISK      │                 |
|  │  (by Quarto)     │  │  (inherent)     │  │  (Quarto-specific)│                 |
|  │                  │  │                 │  │                  │                 |
|  │  XSS-to-RCE     │  │  Code execution │  │  Lua filter RCE  │                 |
|  │  via outputs     │  │  during render  │  │  (os.execute(),  │                 |
|  │  (CVE-2021-     │  │  (no sandbox)   │  │   io.popen())    │                 |
|  │   32797/98)     │  │                 │  │                  │                 |
|  │                  │  │  Pandoc vulns   │  │  Extension trust │                 |
|  │  Server-side    │  │  (CVE-2023-     │  │  (Y/n prompt only│                 |
|  │  attack surface │  │   35936/38745,  │  │   no code signing│                 |
|  │  (no kernel     │  │   CVE-2025-     │  │   no permissions)│                 |
|  │   server)       │  │   51591)        │  │                  │                 |
|  │                  │  │                 │  │  _freeze/ tamper │                 |
|  │  JSON metadata  │  │  Supply chain   │  │  (committed JSON │                 |
|  │  injection      │  │  (Python/R      │  │   not signed)    │                 |
|  │  (plain text    │  │   packages)     │  │                  │                 |
|  │   .qmd format)  │  │                 │  │  CVE-2024-38526  │                 |
|  │                  │  │  PDF/LaTeX      │  │  (polyfill.io in │                 |
|  │  Google Caja    │  │  shell escape   │  │   MathJax CDN)   │                 |
|  │  dependency     │  │  (CVE-2023-     │  │                  │                 |
|  │                  │  │   32700)        │  │  Deno/JS chain   │                 |
|  │  Hidden state   │  │                 │  │  (npm supply     │                 |
|  │  (no stored     │  │  Reproducibility│  │   chain attacks)  │                 |
|  │   cell outputs) │  │  vs security    │  │                  │                 |
|  │                  │  │  tradeoff       │  │  No SECURITY.md  │                 |
|  └─────────────────┘  └─────────────────┘  └─────────────────┘                 |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  DEPENDENCY CHAIN (ATTACK SURFACE)                                              |
|  ──────────────────────────────────                                             |
|                                                                                 |
|  .qmd file                                                                      |
|    |                                                                            |
|    +-- Quarto CLI ──── Pandoc (CVE-2023-35936, CVE-2025-51591)                 |
|    |                      |                                                     |
|    |                      +── Lua filters (os.execute(), io.popen())            |
|    |                      +── LuaTeX (CVE-2023-32700: shell escape bypass)      |
|    |                                                                            |
|    +-- Deno (CVE-2022-24783, CVE-2024-34346: sandbox bypasses)                 |
|    |                                                                            |
|    +-- Jupyter kernel (Python/R/Julia -- full system access)                    |
|    |                                                                            |
|    +-- Extensions (Lua + JS, Y/n trust only)                                   |
|    |                                                                            |
|    +-- CDN deps (polyfill.io CVE-2024-38526, MathJax, etc.)                   |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  OUR MITIGATIONS                                                                |
|  ─────────────────                                                              |
|                                                                                 |
|  | Vector                | Our Defense                                 |       |
|  |-----------------------|---------------------------------------------|       |
|  | Untrusted .qmd render | error:false halts on errors; CI uses freeze |       |
|  | Lua filter injection  | No third-party extensions installed         |       |
|  | _freeze/ tampering    | Code review on all _freeze/ changes         |       |
|  | Supply chain (CDN)    | Quarto >= 1.4.557 (polyfill.io patched)     |       |
|  | Supply chain (Python) | uv.lock pinning + Dependabot alerts         |       |
|  | Sensitive data leak   | AST scanner for PLR codes + abs paths       |       |
|  | CI privilege escal.   | permissions: contents: read in workflows     |       |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| "FIXED" column | `healthy_normal` | Green -- issues eliminated by Quarto |
| "PERSISTS" column | `highlight_accent` | Gold/amber -- still present, inherent to code execution |
| "NEW RISK" column | `abnormal_warning` | Red -- Quarto-introduced attack surface |
| Dependency chain | `primary_pathway` | Blue -- architecture diagram |
| CVE labels | `abnormal_warning` | Red -- specific vulnerability callouts |
| Mitigation table | `healthy_normal` | Green -- defensive measures |
| Background | `background` | Off-white #FBF9F3 |

## Content Elements

1. **Three-column comparison**: FIXED / PERSISTS / NEW (green/gold/red gradient)
2. **FIXED column**: XSS-to-RCE elimination, no server surface, no stored outputs, no Caja, no hidden state
3. **PERSISTS column**: Unsandboxed code execution, Pandoc CVEs, supply chain, PDF shell escape, reproducibility-security tradeoff
4. **NEW RISK column**: Lua filter RCE, weak extension trust, _freeze/ tampering, polyfill.io CVE, Deno/JS chain, no SECURITY.md
5. **Dependency chain diagram**: .qmd → Quarto CLI → Pandoc → Lua/LuaTeX → Deno → kernel → extensions → CDN
6. **CVE annotations**: Each vulnerability labeled with CVE number and year
7. **Our mitigations table**: How Foundation PLR defends against each vector

## Anti-Hallucination Rules

- CVE-2024-38526 is the ONLY documented Quarto-specific CVE. DO NOT invent additional Quarto CVEs.
- Quarto has ZERO published security advisories on GitHub (verified at quarto-dev/quarto-cli/security).
- Quarto has NO SECURITY.md file. This is a fact, not a judgment.
- Pandoc's --sandbox flag does NOT restrict Lua filters. This is documented in the Pandoc manual.
- The polyfill.io fix is in Quarto >= 1.4.557 / 1.5.24 RC. DO NOT claim older versions are safe.
- Lua filters CAN call os.execute() and io.popen(). This is documented Pandoc behavior, not a bug.
- DO NOT claim Quarto is "insecure" or "more dangerous than Jupyter." The security tradeoffs differ.
- DO NOT show AUROC or model performance. This is about infrastructure security.
- The 4.03% reproducibility stat is from Pimentel et al. 2019 (MSR), not from a security study.
- CVE-2025-51591 (Pandoc SSRF) was discovered in September 2025. This is a Pandoc vulnerability, not Quarto-specific.
- quarto-live (WebAssembly execution) is a separate extension, not core Quarto functionality.

## Text Content

### Title Text
"Quarto Security: What Changed, What Didn't"

### Caption
Moving from Jupyter to Quarto eliminates stored-output XSS attacks (CVE-2021-32797/98), server-side attack surface, and JSON metadata injection -- all consequences of Quarto's plain-text `.qmd` format and static HTML output. However, code execution during `quarto render` remains unsandboxed, Pandoc vulnerabilities propagate through Quarto (CVE-2023-35936, CVE-2025-51591), and Quarto introduces new vectors: Lua filters with full `os.execute()` access, a weak extension trust model (Y/n prompt only), `_freeze/` JSON files without integrity verification, and the only real-world Quarto CVE -- the polyfill.io supply chain compromise (CVE-2024-38526) affecting all HTML output. Foundation PLR mitigates these through freeze-first CI, no third-party extensions, AST-based code scanning, and locked dependencies.

## Prompts for Nano Banana Pro

### Style Prompt
Three-column comparison layout on warm off-white background (#FBF9F3). Left column "FIXED" in green (#5B8C3E), center column "PERSISTS" in gold (#D4A03C), right column "NEW RISK" in red (#C44536). Dependency chain as tree diagram below. Professional medical illustration style, no sci-fi. Clean typography with CVE numbers as small badges.

### Content Prompt
Create "Quarto Security: What Changed, What Didn't" three-column comparison:

**TOP -- Three Columns**:
- LEFT (green, "FIXED"): XSS-to-RCE (CVE-2021-32797), server attack surface (no kernel server), JSON metadata injection (plain text format), Google Caja (no sanitizer needed), hidden state (no stored outputs)
- CENTER (gold, "PERSISTS"): Unsandboxed code execution, Pandoc CVEs (CVE-2023-35936, CVE-2025-51591), supply chain (Python/R packages), PDF/LaTeX shell escape (CVE-2023-32700), reproducibility vs security tradeoff
- RIGHT (red, "NEW RISK"): Lua filter RCE (os.execute), extension trust model (Y/n only, no code signing), _freeze/ tampering (unsigned JSON), polyfill.io CVE-2024-38526, Deno/JS dependency chain, no SECURITY.md

**MIDDLE -- Dependency Chain**:
Tree: .qmd → Quarto CLI → {Pandoc (+ Lua + LuaTeX), Deno, Jupyter kernel, Extensions, CDN deps}
Each node annotated with relevant CVEs

**BOTTOM -- Our Mitigations**:
Table mapping vectors to defenses: freeze-first CI, no extensions, code review on _freeze/, uv.lock pinning, AST scanner, workflow permissions

## Alt Text

Three-column security comparison for Quarto notebooks. Left column (green, "FIXED by Quarto"): eliminates stored-output XSS-to-RCE (CVE-2021-32797/98), removes server-side attack surface (no kernel server), eliminates JSON metadata injection via plain-text .qmd format, removes Google Caja dependency, eliminates hidden execution state. Center column (gold, "PERSISTS"): unsandboxed code execution during quarto render, inherited Pandoc vulnerabilities (CVE-2023-35936 arbitrary file write, CVE-2025-51591 SSRF), Python/R supply chain risks, PDF/LaTeX shell escape (CVE-2023-32700), fundamental reproducibility-versus-security tradeoff. Right column (red, "NEW RISKS"): Lua filter arbitrary code execution via os.execute() and io.popen(), weak extension trust model with only Y/n prompt and no code signing or permissions system, unsigned _freeze/ JSON files vulnerable to tampering, polyfill.io CDN supply chain compromise (CVE-2024-38526, the only documented Quarto-specific CVE), expanded Deno/JavaScript/npm dependency chain, and no SECURITY.md file or formal vulnerability disclosure process. Below: dependency chain tree showing attack surface from .qmd file through Quarto CLI to Pandoc, Lua, LuaTeX, Deno, Jupyter kernel, extensions, and CDN dependencies. Bottom: Foundation PLR mitigations table.

## References

- CVE-2024-38526 (polyfill.io): [Posit Support Article](https://support.posit.co/hc/en-us/articles/24767859071895-PolyFill-CDN-Supply-Chain-Attack-CVE-2024-38526)
- CVE-2023-35936/38745 (Pandoc file write): [GitHub Advisory](https://github.com/jgm/pandoc/security/advisories/GHSA-xj5q-fv23-575g)
- CVE-2025-51591 (Pandoc SSRF): [The Hacker News](https://thehackernews.com/2025/09/hackers-exploit-pandoc-cve-2025-51591.html)
- CVE-2023-32700 (LuaTeX shell escape): [TUG Security Page](https://tug.org/~mseven/luatex.html)
- Deno security advisories: [GitHub](https://github.com/denoland/deno/security/advisories)
- Pandoc Lua filter capabilities: [Pandoc Manual](https://pandoc.org/lua-filters.html)
- Pandoc --sandbox limitations: [Pandoc Manual](https://pandoc.org/MANUAL.html)
- Quarto extension management: [quarto.org](https://quarto.org/docs/extensions/managing.html)
- Quarto execution options: [quarto.org](https://quarto.org/docs/projects/code-execution.html)
- quarto-cli security tab: [GitHub](https://github.com/quarto-dev/quarto-cli/security)
- Lua filter RCE (go-pandoc): [Pulse Security](https://pulsesecurity.co.nz/advisories/gopandoc-filter-rce)
- quarto-live (WebAssembly sandbox): [r-wasm.github.io](https://r-wasm.github.io/quarto-live/)
- Jiang et al. 2025 (Jupyter CVE trends): [arXiv:2507.18833](https://arxiv.org/abs/2507.18833)

## Related Figures

- **fig-nb-06**: MLSecOps for Notebooks (Jupyter-focused attack surface map -- COMPANION figure)
- **fig-nb-01**: Notebook Landscape (Jupyter vs Quarto vs Marimo comparison)
- **fig-nb-05**: Notebook Testing Landscape (testing as security layer)
- **fig-repro-03**: Five Horsemen of Irreproducibility (broader context)

## Cross-References

Reader flow: **fig-nb-06** (Jupyter attack surface) → **THIS FIGURE** (what Quarto fixes/doesn't) → **fig-nb-05** (testing as defense) → **fig-nb-03** (our architecture mitigations)

## Status

- [x] Draft created
- [ ] Generated via Nano Banana Pro
- [ ] Placed in notebooks/README.md
