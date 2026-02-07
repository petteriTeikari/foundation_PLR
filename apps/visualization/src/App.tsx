/**
 * Foundation PLR Visualization - Development App
 *
 * Preview and test figures during development.
 * Load real JSON data from the figures/generated/data directory.
 */

import React, { useEffect, useState } from 'react';
import { RetentionCurve } from './components/figures/RetentionCurve';
import { RetentionFigureSchema, normalizeRetentionData, type RetentionData } from './types/figures';

// Metric display labels (mirrors the component's labels)
const METRIC_LABELS: Record<string, string> = {
  auroc: 'AUROC',
  brier: 'Brier Score',
  scaled_brier: 'Scaled Brier',
  net_benefit: 'Net Benefit',
};

// Path to real data (relative to public or fetched from parent directory)
const DATA_PATH = '../figures/generated/data/fig_retained_auroc.json';

function App() {
  const [data, setData] = useState<RetentionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const response = await fetch(DATA_PATH);
        if (!response.ok) {
          throw new Error(`Failed to load: ${response.status}`);
        }

        const rawJson = await response.json();

        // Validate with Zod
        const parsed = RetentionFigureSchema.safeParse(rawJson);
        if (!parsed.success) {
          throw new Error(`Invalid data format: ${parsed.error.message}`);
        }

        // Normalize to consistent format
        const normalized = normalizeRetentionData(parsed.data);
        setData(normalized);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        // Fall back to mock data for development
        setData(generateMockData());
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '2rem', fontFamily: 'var(--font-body)' }}>
        Loading data...
      </div>
    );
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 900,
          fontSize: '2rem',
          letterSpacing: '-0.02em',
          margin: 0,
        }}>
          Foundation PLR Visualizations
        </h1>
        <p style={{
          fontFamily: 'var(--font-body)',
          color: '#666',
          marginTop: '0.5rem',
        }}>
          Interactive preview of publication figures
        </p>
      </header>

      {error && (
        <div style={{
          padding: '1rem',
          background: '#fff3cd',
          border: '1px solid #ffc107',
          borderRadius: '4px',
          marginBottom: '1rem',
          fontFamily: 'var(--font-body)',
        }}>
          <strong>Note:</strong> Using mock data. {error}
        </div>
      )}

      {data && (
        <section>
          <h2 style={{
            fontFamily: 'var(--font-heading)',
            fontWeight: 900,
            fontSize: '1.25rem',
            marginBottom: '1rem',
          }}>
            Retention Curve ({data.metric ? METRIC_LABELS[data.metric] ?? data.metric.toUpperCase() : 'Metric'})
          </h2>

          <div style={{
            background: 'white',
            border: '1px solid #e0e0e0',
            borderRadius: '8px',
            padding: '1rem',
          }}>
            <RetentionCurve
              data={data}
              title={`${METRIC_LABELS[data.metric] ?? data.metric.toUpperCase()} vs Retention Rate`}
              showBaseline={true}
              showAnnotation={true}
              showGrid={true}
              interactive={true}
            />
          </div>

          <div style={{ marginTop: '2rem' }}>
            <h3 style={{
              fontFamily: 'var(--font-body)',
              fontWeight: 500,
              fontSize: '1rem',
              marginBottom: '0.5rem',
            }}>
              Data Summary
            </h3>
            <pre style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              background: '#f5f5f5',
              padding: '1rem',
              borderRadius: '4px',
              overflow: 'auto',
            }}>
{JSON.stringify({
  metric: data.metric,
  n_samples: data.y_true.length,
  retention_range: [
    Math.min(...data.retention_rates).toFixed(2),
    Math.max(...data.retention_rates).toFixed(2),
  ],
  metric_range: [
    Math.min(...data.metric_values).toFixed(3),
    Math.max(...data.metric_values).toFixed(3),
  ],
}, null, 2)}
            </pre>
          </div>
        </section>
      )}
    </div>
  );
}

// Mock data generator for development
function generateMockData(): RetentionData {
  const n = 50;
  const retention_rates = Array.from({ length: n }, (_, i) => 0.1 + (i / (n - 1)) * 0.9);

  // Simulated AUROC that improves as retention decreases
  const metric_values = retention_rates.map(r => {
    const noise = (Math.random() - 0.5) * 0.02;
    return Math.min(0.95, 0.5 + (1 - r) * 0.4 + noise);
  });

  return {
    y_true: Array.from({ length: 200 }, () => Math.random() > 0.7 ? 1 : 0),
    y_prob: Array.from({ length: 200 }, () => Math.random()),
    uncertainty: Array.from({ length: 200 }, () => Math.random() * 0.2),
    retention_rates,
    metric_values,
    metric: 'auroc',
    threshold: null,
  };
}

export default App;
