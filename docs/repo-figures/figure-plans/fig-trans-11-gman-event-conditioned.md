# fig-trans-11: GMAN for Event-Conditioned Time Series

**Status**: 📋 PLANNED
**Tier**: 3 - Alternative Approaches
**Target Persona**: Humanitarian/logistics professionals, EHR data scientists

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-11 |
| Type | Architecture + example diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Introduce GMAN (Graph Mixing Additive Networks) as the appropriate approach for sparse, irregular, event-driven time series where TSFMs fail - specifically for humanitarian logistics, EHRs, and multimodal data.

---

## 3. Key Message

> "When your anomalies are real events (missile strikes, holidays, patient non-visits) rather than measurement errors, you don't want to impute them away. GMAN conditions predictions on external knowledge, treating each trajectory as a graph and each event as a node. It's interpretable, handles irregularity natively, and explains WHY an anomaly occurred."

---

## 4. Literature Sources

| Source | Key Finding |
|--------|-------------|
| Bechler-Speicher et al. 2025 | GMAN: "flexible, interpretable framework for sets of sparse time-series" |
| GMAN Paper | AUROC 76.64% on PhysioNet mortality (beats Transformers, GRU-D, RAINDROP) |
| GMAN Paper | 97.34% accuracy on GossipCop fake news detection |
| GMAN Architecture | ExtGNAN + DeepSet aggregation for variable-length trajectories |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  GMAN: Graph Mixing Additive Networks                                      │
│  For Event-Conditioned Sparse Time Series                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PROBLEM: Logistics Time Series with External Events                   │
│  ───────────────────────────────────────────────────────                   │
│                                                                            │
│  Supply Volume                                                             │
│       │                                                                    │
│    100├─────────┐                                                          │
│       │         │                                                          │
│     50├─────────┼─────────┬─────────┐                                      │
│       │         │    ❌   │         │                                      │
│      0├─────────┴─────────┴─────────┴─────────────────                     │
│       │    Week 1    Week 2    Week 3    Week 4                            │
│                        ↑                                                   │
│                   Missile strike                                           │
│                   (external event)                                         │
│                                                                            │
│  ✗ TSFM approach: Impute Week 2 → WRONG (invents fake deliveries)         │
│  ✓ GMAN approach: Condition on event → EXPLAINS the anomaly               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GMAN ARCHITECTURE                                                         │
│  ────────────────                                                          │
│                                                                            │
│  Input: Set of sparse trajectories + external events                       │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Trajectory 1        Trajectory 2        Trajectory 3           │       │
│  │  (Warehouse A)       (Warehouse B)       (External Events)      │       │
│  │                                                                 │       │
│  │  ●──●──○──●          ●──○──●──●          🔴 Missile strike      │       │
│  │  t1 t2 t3 t4         t1 t2 t3 t4         🟡 Holiday             │       │
│  │      ↓                   ↓                    ↓                 │       │
│  │  ┌───────────────────────────────────────────────────────┐     │       │
│  │  │              ExtGNAN per trajectory                    │     │       │
│  │  │  • Distance function ρ(Δt) for time gaps               │     │       │
│  │  │  • Feature shape functions ψ(x) per variable           │     │       │
│  │  │  • Additive aggregation preserves interpretability     │     │       │
│  │  └───────────────────────────────────────────────────────┘     │       │
│  │                           ↓                                     │       │
│  │  ┌───────────────────────────────────────────────────────┐     │       │
│  │  │              DeepSet Aggregation                       │     │       │
│  │  │  • Permutation-invariant over trajectories             │     │       │
│  │  │  • Non-linear mixing in grouped subsets                │     │       │
│  │  │  • Linear combination of subset representations        │     │       │
│  │  └───────────────────────────────────────────────────────┘     │       │
│  │                           ↓                                     │       │
│  │  ┌───────────────────────────────────────────────────────┐     │       │
│  │  │              Prediction + Interpretation               │     │       │
│  │  │  • Node importance: which measurements matter?         │     │       │
│  │  │  • Graph importance: which warehouses/events?          │     │       │
│  │  │  • Actionable explanation for decision-makers          │     │       │
│  │  └───────────────────────────────────────────────────────┘     │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHY GMAN vs TSFM for Logistics/EHR                                        │
│                                                                            │
│  ┌────────────────────┬──────────────────────────────────────────────┐    │
│  │ Challenge          │ TSFM                │ GMAN                   │    │
│  ├────────────────────┼─────────────────────┼────────────────────────┤    │
│  │ Irregular sampling │ Assumes regular     │ Native handling        │    │
│  │ External events    │ Ignores or imputes  │ Conditions on events   │    │
│  │ Interpretability   │ Black box           │ Node/graph attribution │    │
│  │ Multimodal         │ Separate encoders   │ Unified graph repr.    │    │
│  │ Missing = info     │ Imputes over it     │ Encodes as feature     │    │
│  └────────────────────┴─────────────────────┴────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Content Elements

### Top Section: The Problem
- Supply chain time series with an event-caused gap
- External event annotation (missile strike)
- Contrast: TSFM imputes (WRONG) vs GMAN explains (RIGHT)

### Middle Section: Architecture
- Three example trajectories (two data, one events)
- ExtGNAN block with key components
- DeepSet aggregation
- Prediction with interpretability outputs

### Bottom Section: Comparison Table
- Side-by-side TSFM vs GMAN on key challenges
- Clear winner indication for sparse/event-driven data

---

## 7. Text Content

### Title
"GMAN: Graph Mixing Additive Networks for Event-Conditioned Time Series"

### Caption
"When time series anomalies represent real events (supply disruptions, missile strikes, patient non-visits) rather than measurement errors, imputation destroys valuable information. GMAN (Bechler-Speicher et al. 2025) treats each sparse trajectory as a directed graph and conditions predictions on external events. It provides interpretable attributions at node, graph, and feature levels - explaining WHY an anomaly occurred rather than smoothing it away. On PhysioNet ICU mortality prediction, GMAN achieves 76.64% AUROC, outperforming Transformers (65.1%), GRU-D (67.2%), and specialized models like RAINDROP (72.1%)."

### Callout Box
"Key Insight: GMAN uses a partition of graph subsets to trade off interpretability and expressivity. Singleton subsets give fine-grained node-level importance. Larger subsets enable non-linear mixing for complex interactions. The user controls this trade-off based on their needs."

---

## 8. Nano Banana Pro Prompts

### Primary Prompt
```
Create a professional architecture diagram titled "GMAN: Graph Mixing Additive Networks for Event-Conditioned Time Series"

TOP SECTION - The Problem:
- Show a supply chain time series with a gap in Week 2
- Annotate the gap with "🔴 Missile strike (external event)"
- Two comparison boxes:
  - "TSFM: Imputes Week 2" with ✗ "Invents fake deliveries"
  - "GMAN: Conditions on event" with ✓ "Explains the anomaly"

MIDDLE SECTION - Architecture:
- Three parallel input trajectories as directed graphs (nodes connected by edges)
- Trajectory labels: "Warehouse A", "Warehouse B", "External Events"
- Flow diagram showing:
  1. ExtGNAN block (distance function, feature shape functions)
  2. DeepSet Aggregation block
  3. Prediction + Interpretation block

BOTTOM SECTION - Comparison Table:
Table comparing TSFM vs GMAN on:
- Irregular sampling
- External events
- Interpretability
- Multimodal data
- Missing = information

Style:
- Clean, academic diagram
- Graph nodes as circles with edges
- Blue for data, red for events, green for outputs
- Sans-serif fonts, high information density
```

### Refinement Prompt
```
Refine to add:

1. Mathematical notation in ExtGNAN block:
   - "ρ(Δt): time distance function"
   - "ψ_l(x): feature shape functions"
   - "h_j = Σ_w ρ(Δ(w,j)) · ψ(x_w)"

2. Performance callout:
   - "PhysioNet Mortality: 76.64% AUROC"
   - "vs Transformer: 65.1%"
   - "vs GRU-D: 67.2%"

3. Interpretability output examples:
   - "Node importance: Measurement on Day 3 was critical"
   - "Graph importance: Warehouse B trajectory matters most"
   - "Event importance: Missile strike explains Week 2 drop"

4. Subtle humanitarian context (UN WFP style)
```

---

## 9. Alt Text

"An architecture diagram for GMAN (Graph Mixing Additive Networks). Top section shows a supply chain time series with a gap caused by a missile strike, contrasting TSFM (which would impute and invent fake data) with GMAN (which conditions on the event and explains the anomaly). Middle section shows the GMAN architecture: three input trajectories represented as directed graphs flow through ExtGNAN processing (with distance and shape functions), then DeepSet aggregation, then prediction with interpretable outputs. Bottom section is a comparison table showing GMAN advantages over TSFMs for irregular sampling, external events, interpretability, multimodal data, and treating missingness as information."

---

## 10. Related Figures

- fig-trans-01: When NOT to Impute (the problem GMAN solves)
- fig-trans-12: M-GAM: Missing Values as Features (related approach)
- fig-trans-03: Domain Fit Matrix (where GMAN applies)

---

## 11. Validation Checklist

- [ ] GMAN architecture accurately represented?
- [ ] Performance numbers correct (76.64% AUROC)?
- [ ] Comparison fair to TSFMs?
- [ ] Humanitarian logistics context appropriate?
- [ ] Mathematical notation correct?
- [ ] Colorblind safe?
- [ ] Alt text complete?

---

*Last updated: 2026-02-01*
