// ESLint Configuration for Foundation PLR Visualization
// =====================================================
// Uses new flat config format (ESLint 9+)
//
// Addresses: GAP-06 (TypeScript ESLint missing)
// Created: 2026-01-29

import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactPlugin from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  // Base JavaScript rules
  js.configs.recommended,

  // TypeScript recommended rules
  ...tseslint.configs.recommended,

  // Project-specific configuration
  {
    files: ["**/*.ts", "**/*.tsx"],
    plugins: {
      react: reactPlugin,
      "react-hooks": reactHooks,
    },
    languageOptions: {
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      // =============================================================================
      // ANTI-HARDCODING RULES (Critical for reproducibility)
      // =============================================================================

      // Warn on magic numbers (but allow common array indices and -1/0/1)
      "no-magic-numbers": [
        "warn",
        {
          ignore: [-1, 0, 1, 2],
          ignoreArrayIndexes: true,
          enforceConst: true,
          detectObjects: false,
        },
      ],

      // =============================================================================
      // TYPESCRIPT RULES
      // =============================================================================

      // Allow unused vars prefixed with underscore
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
        },
      ],

      // Allow explicit any in specific cases (warn, not error)
      "@typescript-eslint/no-explicit-any": "warn",

      // Require type annotations for public API
      "@typescript-eslint/explicit-function-return-type": "off",

      // =============================================================================
      // REACT RULES
      // =============================================================================

      // React hooks rules
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // React 17+ JSX transform (no need to import React)
      "react/react-in-jsx-scope": "off",

      // =============================================================================
      // STYLE RULES (D3 + SVG patterns)
      // =============================================================================

      // Enforce consistent naming
      "@typescript-eslint/naming-convention": [
        "warn",
        {
          selector: "variable",
          format: ["camelCase", "UPPER_CASE", "PascalCase"],
          leadingUnderscore: "allow",
        },
        {
          selector: "function",
          format: ["camelCase", "PascalCase"],
        },
        {
          selector: "typeLike",
          format: ["PascalCase"],
        },
      ],
    },
  },

  // Ignore patterns
  {
    ignores: ["dist/**", "node_modules/**", "*.config.js", "*.config.ts"],
  }
);
