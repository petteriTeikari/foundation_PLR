# fig-nb-06: MLSecOps for Notebooks (Attack Surface Map)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-06 |
| **Title** | MLSecOps for Notebooks (Attack Surface Map) |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | ML Engineer, Security Engineer |
| **Location** | docs/notebooks-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Map the notebook-specific attack surface as a threat model diagram, showing six distinct attack vectors, real CVEs, and the defense layers this project implements. Raise security awareness for ML practitioners who treat notebooks as safe sandboxes.

## Key Message

"Jupyter notebooks are a growing attack surface: NVD reports doubled between 2023 and 2024, with vectors ranging from XSS-to-RCE in rendered outputs to pickle deserialization bypasses and supply chain attacks targeting exposed instances."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| CVE-2021-32797/98 | XSS to RCE via Caja sandbox bypass in notebook outputs | [NVD](https://nvd.nist.gov/vuln/detail/CVE-2021-32797) |
| CVE-2025-1716 | picklescan bypass; JFrog reports 89% evasion rate | [JFrog Security Research](https://jfrog.com/blog/picklescan-bypass/) |
| Qubitstrike 2023 | First Jupyter-targeted cryptomining campaign | [Cado Security](https://www.cadosecurity.com/qubitstrike/) |
| Panamorfi 2024 | DDoS campaign via exposed Jupyter instances | [Aqua Security](https://www.aquasec.com/blog/panamorfi/) |
| OWASP ML Top 10 | ML06: AI Supply Chain Attacks | [owasp.org/www-project-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/) |
| NB Defense | Protect AI notebook security scanner | [github.com/protectai/nbdefense](https://github.com/protectai/nbdefense) |

## Visual Concept

```
+---------------------------------------------------------------------------------+
|                    MLSecOps FOR NOTEBOOKS                                        |
|                    Attack Surface Map                                            |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  THREAT MODEL: 6 ATTACK VECTORS                                                |
|  ================================                                               |
|                                                                                 |
|                       [2] PICKLE                                                |
|                       DESERIALIZATION                                           |
|                       CVE-2025-1716                                             |
|                       89% picklescan                                            |
|                       evasion (JFrog)                                           |
|                            |                                                    |
|       [1] MALICIOUS        |        [3] HIDDEN STATE                            |
|       OUTPUTS              |        Credentials in                               |
|       CVE-2021-32797/98    |        JSON metadata                                |
|       XSS to RCE via       |        survive cell                                 |
|       Caja bypass          |        clearing                                     |
|            \               |              /                                      |
|             \              |             /                                       |
|              \             |            /                                        |
|               +------ [NOTEBOOK] ------+                                        |
|              /             |            \                                        |
|             /              |             \                                       |
|            /               |              \                                      |
|       [6] EXPOSED          |        [4] UNSANDBOXED                              |
|       INSTANCES            |        RENDER                                       |
|       Crypto mining,       |        quarto/nbconvert                             |
|       NTLM hash leaks     |        = full code exec,                            |
|       (Windows)            |        no sandbox                                   |
|                            |                                                    |
|                       [5] SUPPLY CHAIN                                          |
|                       Qubitstrike 2023                                          |
|                       Panamorfi DDoS 2024                                       |
|                       First Jupyter                                             |
|                       ransomware 2024                                           |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  CVE TIMELINE                                                                   |
|  ============                                                                   |
|                                                                                 |
|  2021  |||               CVE-2021-32797/98 (XSS->RCE)                           |
|  2022  ||||              Notebook metadata credential leaks                     |
|  2023  |||||||           Qubitstrike, exposed instance campaigns                 |
|  2024  ||||||||||||||    Panamorfi DDoS, ransomware, CVE doubling               |
|  2025  |||||||||||       CVE-2025-1716 picklescan bypass                        |
|                                                                                 |
|  KEY STAT: "NVD Jupyter reports doubled between 2023-2024"                      |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  OUR DEFENSE LAYERS                                                             |
|  ==================                                                             |
|                                                                                 |
|  | Attack Vector       | Defense                        | Implementation     | |
|  |---------------------|--------------------------------|--------------------| |
|  | Malicious outputs   | No untrusted notebooks         | Code review + CI   | |
|  | Pickle deser.       | No pickle in notebooks         | DuckDB for data    | |
|  | Hidden state        | check_notebook_format.py       | Pre-commit hook    | |
|  | Unsandboxed render  | Trusted code only in CI        | GitHub Actions     | |
|  | Supply chain        | uv.lock pinning                | Lockfile in git    | |
|  | Exposed instances   | No public Jupyter server       | Local dev only     | |
|                                                                                 |
|  TOOLS: NB Defense (Protect AI), ModelScan, Sigstore model signing              |
|  OWASP: ML06 - AI Supply Chain Attacks                                          |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| Central notebook node | `primary_pathway` | Deep blue focal point |
| Attack vectors (1-6) | `abnormal_warning` | Red - threat indicators |
| CVE labels | `abnormal_warning` | Red - vulnerability callouts |
| Defense table rows | `healthy_normal` | Green - mitigation active |
| CVE timeline bars | `highlight_accent` | Gold - temporal emphasis |
| Neutral text | `secondary_pathway` | Gray body text |

## Content Elements

1. **Central threat diagram**: Notebook node with 6 radiating attack vectors
2. **CVE callouts**: Specific CVE numbers with brief descriptions per vector
3. **CVE timeline bar chart**: Horizontal bars showing growth 2021-2025
4. **Key statistic callout**: NVD doubling between 2023-2024
5. **Defense table**: Attack vector mapped to our specific mitigation
6. **Tool references**: NB Defense, ModelScan, Sigstore
7. **OWASP reference**: ML06 Supply Chain Attacks classification

## Anti-Hallucination Rules

- DO NOT invent CVE numbers. Use only: CVE-2021-32797, CVE-2021-32798, CVE-2025-1716.
- DO NOT fabricate exact NVD counts. The "doubled" claim is directional, from public NVD query trends.
- DO NOT claim we run NB Defense or ModelScan. They are listed as ecosystem tools, not our stack.
- The JFrog 89% picklescan evasion rate is from their published security research blog post.
- Qubitstrike was discovered by Cado Security in 2023. Panamorfi was reported by Aqua Security in 2024.
- DO NOT show AUROC or any model performance metrics. This figure is about security.
- The "first Jupyter ransomware 2024" refers to reports of ransomware payloads delivered via exposed notebooks (Aqua Security reporting).

## Text Content

### Title Text
"MLSecOps for Notebooks: Attack Surface Map"

### Caption
Jupyter notebooks present six distinct attack vectors: malicious outputs (CVE-2021-32797/98: XSS-to-RCE via Caja sandbox bypass), pickle deserialization (CVE-2025-1716: 89% picklescan evasion per JFrog), hidden state in JSON metadata, unsandboxed rendering, supply chain attacks (Qubitstrike 2023, Panamorfi DDoS 2024), and exposed instances. NVD Jupyter vulnerability reports doubled between 2023 and 2024. Foundation PLR defends via DuckDB (no pickle), pre-commit AST checks, lockfile pinning, and local-only execution.

## Prompts for Nano Banana Pro

### Style Prompt
Threat model diagram with central node and six radiating vectors. Warning red for attack paths. Professional security-focused aesthetic on warm off-white background. CVE timeline as horizontal bar chart. Defense matrix table at bottom. Clean, editorial layout suitable for security documentation.

### Content Prompt
Create "MLSecOps for Notebooks" attack surface map:

**TOP - Threat Model Diagram**:
- Central "Notebook" node in deep blue
- 6 attack vectors radiating outward in red:
  1. Malicious Outputs (CVE-2021-32797/98)
  2. Pickle Deserialization (CVE-2025-1716)
  3. Hidden State (credentials in metadata)
  4. Unsandboxed Render (full code execution)
  5. Supply Chain (Qubitstrike, Panamorfi)
  6. Exposed Instances (crypto mining)

**MIDDLE - CVE Timeline**:
- Horizontal bars 2021-2025 showing growth
- Key stat: "NVD Jupyter reports doubled 2023-2024"

**BOTTOM - Defense Table**:
- Attack vector -> Our defense -> Implementation

## Alt Text

Notebook attack surface threat model. Central Jupyter notebook node with six attack vectors radiating outward: malicious outputs via XSS-to-RCE (CVE-2021-32797/98), pickle deserialization bypass (CVE-2025-1716, 89% evasion rate), hidden credentials in JSON metadata, unsandboxed code execution during render, supply chain attacks (Qubitstrike 2023, Panamorfi DDoS 2024), and exposed notebook instances for crypto mining. CVE timeline shows doubling of NVD Jupyter reports between 2023 and 2024. Defense table maps each vector to project mitigations: DuckDB instead of pickle, AST-based pre-commit checks, lockfile pinning, and local-only execution.

## References

- CVE-2021-32797: Jupyter Notebook XSS to RCE ([NVD](https://nvd.nist.gov/vuln/detail/CVE-2021-32797))
- CVE-2021-32798: JupyterLab XSS to RCE ([NVD](https://nvd.nist.gov/vuln/detail/CVE-2021-32798))
- CVE-2025-1716: picklescan bypass ([JFrog](https://jfrog.com/blog/picklescan-bypass/))
- Qubitstrike: Cado Security 2023 ([cadosecurity.com](https://www.cadosecurity.com/qubitstrike/))
- Panamorfi: Aqua Security 2024 ([aquasec.com](https://www.aquasec.com/blog/panamorfi/))
- NB Defense: Protect AI ([github.com/protectai/nbdefense](https://github.com/protectai/nbdefense))
- ModelScan: Protect AI ([github.com/protectai/modelscan](https://github.com/protectai/modelscan))
- OWASP ML Top 10: ML06 Supply Chain Attacks ([owasp.org](https://owasp.org/www-project-machine-learning-security-top-10/))

## Related Figures

- **fig-nb-05**: Notebook Testing Landscape (testing defenses)
- **fig-repo-08**: Pre-commit Quality Gates (where AST checks run)
- **fig-repro-14**: Lockfiles Time Machine (supply chain defense)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/notebooks-guide.md
