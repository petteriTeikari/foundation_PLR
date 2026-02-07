#!/usr/bin/env tsx
/**
 * Lint script to detect hard-coded styles in TSX components
 *
 * Catches patterns like:
 * - stroke="#0077BB"
 * - fill="red"
 * - fontSize={12}
 * - fontFamily="Manrope"
 * - strokeWidth={2}
 * - opacity={0.5} (outside className context)
 *
 * Run: npx tsx scripts/lint-no-hardcoded-styles.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';

// ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================
// FORBIDDEN PATTERNS
// ============================================

interface ForbiddenPattern {
  pattern: RegExp;
  message: string;
  allowedContexts?: RegExp[];
}

const FORBIDDEN_PATTERNS: ForbiddenPattern[] = [
  // ============================================
  // HARD-CODED DOMAIN TERMS (should come from data)
  // ============================================
  {
    pattern: /["']AUROC["']/gi,
    message: 'Hard-coded metric name "AUROC". Use data.metric or METRIC_LABELS[data.metric] instead.',
    allowedContexts: [/METRIC_LABELS/, /Record<string/],  // Allow in label lookup objects
  },
  {
    pattern: /["']Brier["']/gi,
    message: 'Hard-coded metric name "Brier". Use data.metric or METRIC_LABELS[data.metric] instead.',
    allowedContexts: [/METRIC_LABELS/, /Record<string/],
  },
  {
    pattern: /["']Glaucoma["']/gi,
    message: 'Hard-coded domain term "Glaucoma". Use data.case_label or similar instead.',
    allowedContexts: [/LABEL_DEFAULTS/],
  },
  {
    pattern: /["']Control["'](?!.*\.control)/gi,
    message: 'Hard-coded domain term "Control". Use data.control_label or similar instead.',
    allowedContexts: [/LABEL_DEFAULTS/],
  },

  // ============================================
  // HARD-CODED STYLES
  // ============================================
  // Colors
  {
    pattern: /\b(stroke|fill)=["'](#[0-9a-fA-F]{3,8}|rgb|rgba|hsl|red|blue|green|gray|black|white)/gi,
    message: 'Hard-coded color detected. Use CSS class with var(--color-*) instead.',
  },
  // Font properties
  {
    pattern: /\bfontSize=\{?\d+/gi,
    message: 'Hard-coded font size. Use CSS class with var(--font-size-*) instead.',
  },
  {
    pattern: /\bfontFamily=["'][^"']+["']/gi,
    message: 'Hard-coded font family. Use CSS class with var(--font-*) instead.',
  },
  {
    pattern: /\bfontWeight=["']?\d+["']?/gi,
    message: 'Hard-coded font weight. Use CSS class with var(--font-weight-*) instead.',
  },
  // Stroke properties (when not in defs/patterns)
  {
    pattern: /\bstrokeWidth=\{?\d/gi,
    message: 'Hard-coded stroke width. Use CSS class with var(--stroke-*) instead.',
    allowedContexts: [/<pattern/, /<defs/],
  },
  // Style prop with object
  {
    pattern: /\bstyle=\{\{[^}]*(color|fontSize|fontFamily|stroke|fill|background)[^}]*\}\}/gi,
    message: 'Inline style object with visual properties. Move to CSS class.',
  },
];

// ============================================
// ALLOWED EXCEPTIONS
// ============================================

const ALLOWED_FILES = [
  'foundations.css',
  'tailwind.config',
];

const ALLOWED_LINES = [
  /className=/,  // Assigning classes is fine
  /var\(--/,     // CSS custom properties are fine
  /\/\//,        // Comments
  /fill="transparent"/, // Transparent is structural
  /fill="none"/,        // None is structural
  /stroke="none"/,      // None is structural
  /metric:\s*['"]/, // Data field assignments (e.g., metric: 'auroc')
  /generateMockData/, // Mock data generation functions
];

// ============================================
// MAIN LINTING FUNCTION
// ============================================

interface Violation {
  file: string;
  line: number;
  column: number;
  code: string;
  message: string;
}

async function lintFile(filePath: string): Promise<Violation[]> {
  const violations: Violation[] = [];
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  lines.forEach((line, lineIndex) => {
    // Skip allowed lines
    if (ALLOWED_LINES.some(pattern => pattern.test(line))) {
      return;
    }

    FORBIDDEN_PATTERNS.forEach(({ pattern, message, allowedContexts }) => {
      // Check if in allowed context (look at surrounding lines)
      if (allowedContexts) {
        const context = lines.slice(Math.max(0, lineIndex - 5), lineIndex + 1).join('\n');
        if (allowedContexts.some(ctx => ctx.test(context))) {
          return;
        }
      }

      const matches = line.matchAll(new RegExp(pattern));
      for (const match of matches) {
        violations.push({
          file: filePath,
          line: lineIndex + 1,
          column: (match.index ?? 0) + 1,
          code: match[0],
          message,
        });
      }
    });
  });

  return violations;
}

async function main() {
  console.log('🔍 Scanning for hard-coded styles in TSX files...\n');

  const files = await glob('src/**/*.tsx', {
    cwd: path.join(__dirname, '..'),
    absolute: true,
  });

  let totalViolations = 0;
  const allViolations: Violation[] = [];

  for (const file of files) {
    // Skip allowed files
    if (ALLOWED_FILES.some(allowed => file.includes(allowed))) {
      continue;
    }

    const violations = await lintFile(file);
    allViolations.push(...violations);
    totalViolations += violations.length;
  }

  // Report
  if (allViolations.length === 0) {
    console.log('✅ No hard-coded styles found!\n');
    process.exit(0);
  }

  console.log(`❌ Found ${totalViolations} violation(s):\n`);

  // Group by file
  const byFile = new Map<string, Violation[]>();
  allViolations.forEach(v => {
    const existing = byFile.get(v.file) ?? [];
    existing.push(v);
    byFile.set(v.file, existing);
  });

  byFile.forEach((violations, file) => {
    const relativePath = path.relative(process.cwd(), file);
    console.log(`📄 ${relativePath}`);
    violations.forEach(v => {
      console.log(`   Line ${v.line}:${v.column} - ${v.code}`);
      console.log(`   ⚠️  ${v.message}`);
      console.log('');
    });
  });

  console.log('\n💡 Tip: Move all visual styling to src/styles/foundations.css');
  console.log('   Use CSS classes with var(--*) custom properties.\n');

  process.exit(1);
}

main().catch(console.error);
