import { z } from 'zod';

// ============================================
// Base Schema for All Figures
// ============================================

export const FigureMetadataSchema = z.object({
  title: z.string().optional(),
  generated_at: z.string().optional(),
  git_commit: z.string().optional(),
  python_module: z.string().optional(),
  data_source: z.string().optional(),
});

export const StyleHintsSchema = z.object({
  line_color: z.string().optional(),
  ci_color: z.string().optional(),
  ci_opacity: z.number().optional(),
  baseline_style: z.string().optional(),
});

// ============================================
// Retention Curve Data Schema
// ============================================

export const RetentionDataSchema = z.object({
  y_true: z.array(z.number()),
  y_prob: z.array(z.number()),
  uncertainty: z.array(z.number()),
  retention_rates: z.array(z.number()),
  metric_values: z.array(z.number()),
  metric: z.string(),
  threshold: z.number().nullable(),
});

export const RetentionFigureSchema = z.object({
  $schema: z.string().optional(),
  figure_type: z.literal('retention_curve').optional(),
  metadata: FigureMetadataSchema.optional(),
  style_hints: StyleHintsSchema.optional(),
  // Support both wrapped and unwrapped data
  data: RetentionDataSchema.optional(),
  // Direct fields (current JSON format)
  y_true: z.array(z.number()).optional(),
  y_prob: z.array(z.number()).optional(),
  uncertainty: z.array(z.number()).optional(),
  retention_rates: z.array(z.number()).optional(),
  metric_values: z.array(z.number()).optional(),
  metric: z.string().optional(),
  threshold: z.number().nullable().optional(),
});

// ============================================
// Calibration Curve Data Schema
// ============================================

export const CalibrationDataSchema = z.object({
  y_true: z.array(z.number()),
  y_prob: z.array(z.number()),
  loess_frac: z.number(),
  x_smooth: z.array(z.number()),
  y_smooth: z.array(z.number()),
  x_ci: z.array(z.number()).optional(),
  y_lower: z.array(z.number()).optional(),
  y_upper: z.array(z.number()).optional(),
});

// ============================================
// DCA Curve Data Schema
// ============================================

export const DCADataSchema = z.object({
  thresholds: z.array(z.number()),
  nb_model: z.array(z.number()),
  nb_all: z.array(z.number()),
  nb_none: z.array(z.number()),
  prevalence: z.number(),
  y_true: z.array(z.number()).optional(),
  y_prob: z.array(z.number()).optional(),
});

// ============================================
// TypeScript Types (inferred from Zod)
// ============================================

export type FigureMetadata = z.infer<typeof FigureMetadataSchema>;
export type StyleHints = z.infer<typeof StyleHintsSchema>;
export type RetentionData = z.infer<typeof RetentionDataSchema>;
export type RetentionFigure = z.infer<typeof RetentionFigureSchema>;
export type CalibrationData = z.infer<typeof CalibrationDataSchema>;
export type DCAData = z.infer<typeof DCADataSchema>;

// ============================================
// Dimension Types
// ============================================

export interface Dimensions {
  width: number;
  height: number;
  marginTop: number;
  marginRight: number;
  marginBottom: number;
  marginLeft: number;
  innerWidth: number;
  innerHeight: number;
}

export interface Point {
  x: number;
  y: number;
}

// ============================================
// Helper to normalize data format
// ============================================

export function normalizeRetentionData(raw: RetentionFigure): RetentionData {
  // If data is wrapped in 'data' field
  if (raw.data) {
    return raw.data;
  }

  // Otherwise, extract direct fields
  return {
    y_true: raw.y_true ?? [],
    y_prob: raw.y_prob ?? [],
    uncertainty: raw.uncertainty ?? [],
    retention_rates: raw.retention_rates ?? [],
    metric_values: raw.metric_values ?? [],
    metric: raw.metric ?? 'auroc',
    threshold: raw.threshold ?? null,
  };
}
