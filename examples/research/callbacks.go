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

package main

import (
	"fmt"
	"log"
	"regexp"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// collectResearchSourcesCallback collects and organizes web-based research sources
// and their supported claims from agent events. It processes the agent's session
// events to extract web source details and associated text segments with confidence
// scores. The aggregated source information and a mapping of URLs to short IDs are
// stored in session state.
func collectResearchSourcesCallback(ctx agent.CallbackContext) (*genai.Content, error) {
	state := ctx.State()

	urlToShortID := make(map[string]string)
	if val, err := state.Get("url_to_short_id"); err == nil {
		if m, ok := val.(map[string]string); ok {
			urlToShortID = m
		} else if m, ok := val.(map[string]any); ok {
			for k, v := range m {
				if vs, ok := v.(string); ok {
					urlToShortID[k] = vs
				}
			}
		}
	}

	sources := make(map[string]*SourceInfo)
	if val, err := state.Get("sources"); err == nil {
		if m, ok := val.(map[string]*SourceInfo); ok {
			sources = m
		} else if m, ok := val.(map[string]any); ok {
			for k, v := range m {
				if si, ok := v.(*SourceInfo); ok {
					sources[k] = si
				}
			}
		}
	}

	idCounter := len(urlToShortID) + 1

	roState := ctx.ReadonlyState()
	allState := roState.All()

	for k, v := range allState {
		if k != "_last_grounding_metadata" {
			continue
		}
		gm, ok := v.(*genai.GroundingMetadata)
		if !ok || gm == nil || gm.GroundingChunks == nil {
			continue
		}

		chunksInfo := make(map[int]string)
		for idx, chunk := range gm.GroundingChunks {
			if chunk.Web == nil {
				continue
			}

			url := chunk.Web.URI
			title := chunk.Web.Title
			if title == chunk.Web.Domain {
				title = chunk.Web.Domain
			}

			if _, exists := urlToShortID[url]; !exists {
				shortID := fmt.Sprintf("src-%d", idCounter)
				urlToShortID[url] = shortID
				sources[shortID] = &SourceInfo{
					ShortID:         shortID,
					Title:           title,
					URL:             url,
					Domain:          chunk.Web.Domain,
					SupportedClaims: []SupportedClaim{},
				}
				idCounter++
			}
			chunksInfo[idx] = urlToShortID[url]
		}

		if gm.GroundingSupports != nil {
			for _, support := range gm.GroundingSupports {
				confidenceScores := support.ConfidenceScores
				chunkIndices := support.GroundingChunkIndices
				for i, chunkIdx := range chunkIndices {
					if shortID, exists := chunksInfo[int(chunkIdx)]; exists {
						confidence := 0.5
						if i < len(confidenceScores) {
							confidence = float64(confidenceScores[i])
						}
						textSegment := ""
						if support.Segment != nil {
							textSegment = support.Segment.Text
						}
						sources[shortID].SupportedClaims = append(sources[shortID].SupportedClaims, SupportedClaim{
							TextSegment: textSegment,
							Confidence:  confidence,
						})
					}
				}
			}
		}
	}

	if err := state.Set("url_to_short_id", urlToShortID); err != nil {
		return nil, fmt.Errorf("failed to set url_to_short_id: %w", err)
	}
	if err := state.Set("sources", sources); err != nil {
		return nil, fmt.Errorf("failed to set sources: %w", err)
	}

	return nil, nil
}

var citeTagRegex = regexp.MustCompile(`<cite\s+source\s*=\s*["']?\s*(src-\d+)\s*["']?\s*/>`)
var spacePunctuationRegex = regexp.MustCompile(`\s+([.,;:])`)

// citationReplacementCallback replaces citation tags in a report with Markdown-formatted links.
// It processes 'final_cited_report' from context state, converting tags like
// <cite source="src-N"/> into hyperlinks using source information from session state.
func citationReplacementCallback(ctx agent.CallbackContext) (*genai.Content, error) {
	state := ctx.State()

	finalReport := ""
	if val, err := state.Get("final_cited_report"); err == nil {
		if s, ok := val.(string); ok {
			finalReport = s
		}
	}

	sources := make(map[string]*SourceInfo)
	if val, err := state.Get("sources"); err == nil {
		if m, ok := val.(map[string]*SourceInfo); ok {
			sources = m
		} else if m, ok := val.(map[string]any); ok {
			for k, v := range m {
				if si, ok := v.(*SourceInfo); ok {
					sources[k] = si
				}
			}
		}
	}

	processedReport := citeTagRegex.ReplaceAllStringFunc(finalReport, func(match string) string {
		submatches := citeTagRegex.FindStringSubmatch(match)
		if len(submatches) < 2 {
			return ""
		}
		shortID := submatches[1]
		sourceInfo, exists := sources[shortID]
		if !exists {
			log.Printf("Invalid citation tag found and removed: %s", match)
			return ""
		}
		displayText := sourceInfo.Title
		if displayText == "" {
			displayText = sourceInfo.Domain
		}
		if displayText == "" {
			displayText = shortID
		}
		return fmt.Sprintf(" [%s](%s)", displayText, sourceInfo.URL)
	})

	processedReport = spacePunctuationRegex.ReplaceAllString(processedReport, "$1")

	if err := state.Set("final_report_with_citations", processedReport); err != nil {
		return nil, fmt.Errorf("failed to set final_report_with_citations: %w", err)
	}

	return genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText(processedReport)}, genai.RoleModel), nil
}

// storeGroundingMetadataCallback stores the grounding metadata from the last LLM response
// in session state for later processing by collectResearchSourcesCallback.
func storeGroundingMetadataCallback(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	if llmResponse == nil || llmResponse.GroundingMetadata == nil {
		return llmResponse, llmResponseError
	}

	state := ctx.State()
	if err := state.Set("_last_grounding_metadata", llmResponse.GroundingMetadata); err != nil {
		log.Printf("Failed to store grounding metadata: %v", err)
	}

	return llmResponse, llmResponseError
}

// formatGroundingReferences appends grounding references as markdown to the response.
func formatGroundingReferences(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	if llmResponse == nil || llmResponse.Content == nil || llmResponse.Content.Parts == nil || llmResponse.GroundingMetadata == nil {
		return llmResponse, llmResponseError
	}

	var references []string
	for _, chunk := range llmResponse.GroundingMetadata.GroundingChunks {
		var title, uri, text string
		if chunk.RetrievedContext != nil {
			title = chunk.RetrievedContext.Title
			uri = chunk.RetrievedContext.URI
			text = chunk.RetrievedContext.Text
		} else if chunk.Web != nil {
			title = chunk.Web.Title
			uri = chunk.Web.URI
		}

		var parts []string
		if title != "" {
			parts = append(parts, title)
		}
		if uri != "" {
			parts = append(parts, uri)
		}
		if text != "" {
			parts = append(parts, text)
		}
		if len(parts) > 0 {
			references = append(references, "* "+strings.Join(parts, ": "))
		}
	}

	if len(references) > 0 {
		refText := "\n\nReference:\n\n" + strings.Join(references, "\n")
		llmResponse.Content.Parts = append(llmResponse.Content.Parts, genai.NewPartFromText(refText))
	}

	return llmResponse, llmResponseError
}
