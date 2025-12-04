// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package anthropic

import (
	"testing"
)

func TestNewModel_DirectAPI(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-api-key")
	t.Setenv("ANTHROPIC_USE_VERTEX", "false")

	model, err := NewModel(t.Context(), "claude-sonnet-4-20250514", nil)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	if model.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("Name() = %q, want %q", model.Name(), "claude-sonnet-4-20250514")
	}
}

func TestNewModel_WithConfig(t *testing.T) {
	cfg := &Config{
		APIKey:           "test-api-key",
		DefaultMaxTokens: 2048,
		Variant:          VariantAnthropicAPI,
	}

	model, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	if model.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("Name() = %q, want %q", model.Name(), "claude-sonnet-4-20250514")
	}

	// Verify internal state
	am := model.(*anthropicModel)
	if am.defaultMaxTokens != 2048 {
		t.Errorf("defaultMaxTokens = %d, want %d", am.defaultMaxTokens, 2048)
	}
	if am.variant != VariantAnthropicAPI {
		t.Errorf("variant = %q, want %q", am.variant, VariantAnthropicAPI)
	}
}

func TestNewModel_VertexAI_MissingProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_REGION", "us-central1")

	cfg := &Config{
		Variant: VariantVertexAI,
	}

	_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
	if err == nil {
		t.Fatal("NewModel() expected error for missing project ID")
	}
}

func TestNewModel_VertexAI_MissingRegion(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
	t.Setenv("GOOGLE_CLOUD_REGION", "")

	cfg := &Config{
		Variant: VariantVertexAI,
	}

	_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
	if err == nil {
		t.Fatal("NewModel() expected error for missing region")
	}
}

func TestNewModel_DefaultMaxTokens(t *testing.T) {
	cfg := &Config{
		APIKey:  "test-api-key",
		Variant: VariantAnthropicAPI,
	}

	model, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	am := model.(*anthropicModel)
	if am.defaultMaxTokens != defaultMaxTokens {
		t.Errorf("defaultMaxTokens = %d, want default %d", am.defaultMaxTokens, defaultMaxTokens)
	}
}

func TestGetVariant(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		want     string
	}{
		{"empty", "", VariantAnthropicAPI},
		{"false", "false", VariantAnthropicAPI},
		{"true", "true", VariantVertexAI},
		{"1", "1", VariantVertexAI},
		{"0", "0", VariantAnthropicAPI},
		{"TRUE", "TRUE", VariantVertexAI},
		{"with_whitespace", "  true  ", VariantVertexAI},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("ANTHROPIC_USE_VERTEX", tt.envValue)

			got := GetVariant()
			if got != tt.want {
				t.Errorf("GetVariant() = %q, want %q", got, tt.want)
			}
		})
	}
}
