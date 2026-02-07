/**
 * RetentionCurve - Metric vs Retention Rate Visualization
 *
 * PARAMETRIC DESIGN:
 * - X-axis: Retention rate (from data.retention_rates)
 * - Y-axis: Any metric (from data.metric_values, labeled by data.metric)
 * - Labels: Derived from data, NEVER hard-coded
 *
 * Shows how any model performance metric changes when using uncertainty
 * to selectively classify only the most certain predictions.
 *
 * Works with: AUROC, Brier Score, Scaled Brier, Net Benefit, or any custom metric.
 * The metric type is determined by data.metric field, not by this component.
 *
 * ALL STYLING via CSS classes - no hard-coded values!
 */

import React, { useMemo, useCallback, useState } from 'react';
import * as d3 from 'd3';
import type { RetentionData, Dimensions, Point } from '@/types/figures';

// ============================================
// Props Interface
// ============================================

interface RetentionCurveProps {
  data: RetentionData;
  width?: number;
  height?: number;
  title?: string;
  showBaseline?: boolean;
  showAnnotation?: boolean;
  showGrid?: boolean;
  interactive?: boolean;
  className?: string;
  onHover?: (retention: number | null, metric: number | null) => void;
}

// ============================================
// Layout Configuration (structure, not style)
// ============================================

const LAYOUT = {
  width: 600,
  height: 420,
  margin: { top: 48, right: 32, bottom: 56, left: 64 },
  annotation: { width: 168, height: 56 },
  tickCount: { x: 5, y: 5 },
};

// Format metric names for display
const METRIC_LABELS: Record<string, string> = {
  auroc: 'AUROC',
  brier: 'Brier Score',
  scaled_brier: 'Scaled Brier',
  net_benefit: 'Net Benefit',
};

// ============================================
// Component
// ============================================

export const RetentionCurve: React.FC<RetentionCurveProps> = ({
  data,
  width = LAYOUT.width,
  height = LAYOUT.height,
  title,
  showBaseline = true,
  showAnnotation = true,
  showGrid = true,
  interactive = false,
  className = '',
  onHover,
}) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  // Calculate dimensions
  const dims: Dimensions = useMemo(() => ({
    width,
    height,
    marginTop: LAYOUT.margin.top,
    marginRight: LAYOUT.margin.right,
    marginBottom: LAYOUT.margin.bottom,
    marginLeft: LAYOUT.margin.left,
    innerWidth: width - LAYOUT.margin.left - LAYOUT.margin.right,
    innerHeight: height - LAYOUT.margin.top - LAYOUT.margin.bottom,
  }), [width, height]);

  // Prepare data points
  const points: Point[] = useMemo(() => {
    return data.retention_rates.map((retention, i) => ({
      x: retention,
      y: data.metric_values[i] ?? 0,
    }));
  }, [data]);

  // X Scale
  const xScale = useMemo(() => {
    const minX = Math.min(...data.retention_rates);
    const maxX = Math.max(...data.retention_rates);
    return d3.scaleLinear()
      .domain([minX - 0.02, maxX + 0.02])
      .range([0, dims.innerWidth]);
  }, [data.retention_rates, dims.innerWidth]);

  // Y Scale
  const yScale = useMemo(() => {
    const minY = Math.min(...data.metric_values);
    const maxY = Math.max(...data.metric_values);
    const padding = (maxY - minY) * 0.1 || 0.05;
    return d3.scaleLinear()
      .domain([Math.max(0, minY - padding), Math.min(1, maxY + padding)])
      .range([dims.innerHeight, 0]);
  }, [data.metric_values, dims.innerHeight]);

  // Line path generator
  const linePath = useMemo(() => {
    const lineGen = d3.line<Point>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX);
    return lineGen(points) ?? '';
  }, [points, xScale, yScale]);

  // Baseline value (at 100% retention)
  const baselineValue = useMemo(() => {
    const idx = data.retention_rates.findIndex(r => r >= 0.99);
    return idx >= 0 ? data.metric_values[idx] : data.metric_values[data.metric_values.length - 1];
  }, [data]);

  // Annotation data (improvement at 50% retention)
  const annotationData = useMemo(() => {
    const idx50 = data.retention_rates.findIndex(r => r >= 0.5);
    if (idx50 < 0) return null;

    const metric50 = data.metric_values[idx50] ?? 0;
    const baseline = baselineValue ?? 0;
    const improvement = baseline !== 0
      ? ((metric50 - baseline) / Math.abs(baseline)) * 100
      : 0;

    return {
      retention: data.retention_rates[idx50] ?? 0.5,
      metric: metric50,
      improvement,
    };
  }, [data, baselineValue]);

  // Axis ticks
  const xTicks = useMemo(() =>
    xScale.ticks(LAYOUT.tickCount.x).map(v => ({ value: v, pos: xScale(v) })),
    [xScale]
  );

  const yTicks = useMemo(() =>
    yScale.ticks(LAYOUT.tickCount.y).map(v => ({ value: v, pos: yScale(v) })),
    [yScale]
  );

  // Hover handling
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGRectElement>) => {
    if (!interactive) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const retention = xScale.invert(mouseX);

    // Find closest point
    let closestIdx = 0;
    let minDist = Infinity;
    data.retention_rates.forEach((r, i) => {
      const dist = Math.abs(r - retention);
      if (dist < minDist) { minDist = dist; closestIdx = i; }
    });

    setHoverIndex(closestIdx);
    onHover?.(data.retention_rates[closestIdx] ?? null, data.metric_values[closestIdx] ?? null);
  }, [interactive, xScale, data, onHover]);

  const handleMouseLeave = useCallback(() => {
    setHoverIndex(null);
    onHover?.(null, null);
  }, [onHover]);

  const metricLabel = METRIC_LABELS[data.metric] ?? data.metric.toUpperCase();

  return (
    <svg
      className={`figure-retention-curve ${className}`}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      {/* Title */}
      {title && (
        <text
          x={dims.marginLeft}
          y={28}
          className="figure-title"
        >
          {title}
        </text>
      )}

      {/* Plot area */}
      <g transform={`translate(${dims.marginLeft}, ${dims.marginTop})`}>

        {/* Grid lines */}
        {showGrid && (
          <g className="grid-lines">
            {yTicks.map(t => (
              <line
                key={`gh-${t.value}`}
                x1={0} y1={t.pos}
                x2={dims.innerWidth} y2={t.pos}
                className="grid-line"
              />
            ))}
            {xTicks.map(t => (
              <line
                key={`gv-${t.value}`}
                x1={t.pos} y1={0}
                x2={t.pos} y2={dims.innerHeight}
                className="grid-line"
              />
            ))}
          </g>
        )}

        {/* Baseline reference */}
        {showBaseline && baselineValue !== undefined && (
          <line
            x1={0} y1={yScale(baselineValue)}
            x2={dims.innerWidth} y2={yScale(baselineValue)}
            className="baseline-line"
          />
        )}

        {/* Main data line */}
        <path d={linePath} className="data-line" />

        {/* Data points */}
        <g className="data-points">
          {points.map((pt, i) => (
            <circle
              key={i}
              cx={xScale(pt.x)}
              cy={yScale(pt.y)}
              className={`data-point ${hoverIndex === i ? 'data-point--active' : ''} ${hoverIndex !== null && hoverIndex !== i ? 'data-point--faded' : ''}`}
            />
          ))}
        </g>

        {/* X-Axis */}
        <g className="axis axis-x" transform={`translate(0, ${dims.innerHeight})`}>
          <line x1={0} y1={0} x2={dims.innerWidth} y2={0} className="axis-line" />
          {xTicks.map(t => (
            <g key={`xt-${t.value}`} transform={`translate(${t.pos}, 0)`}>
              <line y1={0} y2={6} className="axis-tick" />
              <text y={22} className="tick-label" textAnchor="middle">
                {Math.round(t.value * 100)}%
              </text>
            </g>
          ))}
          <text
            x={dims.innerWidth / 2}
            y={46}
            className="axis-label"
            textAnchor="middle"
          >
            Retention Rate
          </text>
        </g>

        {/* Y-Axis */}
        <g className="axis axis-y">
          <line x1={0} y1={0} x2={0} y2={dims.innerHeight} className="axis-line" />
          {yTicks.map(t => (
            <g key={`yt-${t.value}`} transform={`translate(0, ${t.pos})`}>
              <line x1={0} x2={-6} className="axis-tick" />
              <text x={-12} className="tick-label" textAnchor="end" dominantBaseline="middle">
                {t.value.toFixed(2)}
              </text>
            </g>
          ))}
          <text
            transform={`translate(-48, ${dims.innerHeight / 2}) rotate(-90)`}
            className="axis-label"
            textAnchor="middle"
          >
            {metricLabel}
          </text>
        </g>

        {/* Annotation box */}
        {showAnnotation && annotationData && (
          <g className="annotation-group" transform={`translate(${dims.innerWidth - LAYOUT.annotation.width - 12}, 12)`}>
            <rect
              width={LAYOUT.annotation.width}
              height={LAYOUT.annotation.height}
              className="annotation-box"
            />
            <text x={12} y={22} className="annotation">
              At 50% retention:
            </text>
            <text x={12} y={40} className="annotation">
              <tspan className="annotation-value">{annotationData.metric.toFixed(3)}</tspan>
              <tspan className="annotation-change" dx={8}>
                ({annotationData.improvement >= 0 ? '+' : ''}{annotationData.improvement.toFixed(1)}%)
              </tspan>
            </text>
          </g>
        )}

        {/* Legend */}
        {showBaseline && (
          <g className="legend" transform={`translate(${dims.innerWidth - LAYOUT.annotation.width - 12}, ${showAnnotation ? 78 : 12})`}>
            <line x1={0} y1={10} x2={24} y2={10} className="baseline-line" />
            <text x={32} y={14} className="legend-label">
              Baseline ({baselineValue?.toFixed(3)})
            </text>
          </g>
        )}

        {/* Interactive overlay */}
        {interactive && (
          <rect
            x={0} y={0}
            width={dims.innerWidth}
            height={dims.innerHeight}
            className="interaction-overlay"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
        )}

        {/* Hover indicator */}
        {interactive && hoverIndex !== null && (
          <line
            x1={xScale(data.retention_rates[hoverIndex] ?? 0)}
            y1={0}
            x2={xScale(data.retention_rates[hoverIndex] ?? 0)}
            y2={dims.innerHeight}
            className="hover-line"
          />
        )}
      </g>
    </svg>
  );
};

export default RetentionCurve;
