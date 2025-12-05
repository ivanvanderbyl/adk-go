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

import "google.golang.org/genai"

// SearchQuery represents a specific search query for web search.
type SearchQuery struct {
	SearchQuery string `json:"search_query"`
}

// Feedback provides evaluation feedback on research quality.
type Feedback struct {
	Grade           string        `json:"grade"`
	Comment         string        `json:"comment"`
	FollowUpQueries []SearchQuery `json:"follow_up_queries,omitempty"`
}

// SourceInfo stores information about a web source.
type SourceInfo struct {
	ShortID         string          `json:"short_id"`
	Title           string          `json:"title"`
	URL             string          `json:"url"`
	Domain          string          `json:"domain"`
	SupportedClaims []SupportedClaim `json:"supported_claims"`
}

// SupportedClaim stores a text segment with its confidence score.
type SupportedClaim struct {
	TextSegment string  `json:"text_segment"`
	Confidence  float64 `json:"confidence"`
}

// FeedbackSchema returns the Gemini schema for the Feedback output.
func FeedbackSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"grade": {
				Type:        genai.TypeString,
				Enum:        []string{"pass", "fail"},
				Description: "Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision.",
			},
			"comment": {
				Type:        genai.TypeString,
				Description: "Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research.",
			},
			"follow_up_queries": {
				Type:        genai.TypeArray,
				Description: "A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'.",
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"search_query": {
							Type:        genai.TypeString,
							Description: "A highly specific and targeted query for web search.",
						},
					},
					Required: []string{"search_query"},
				},
			},
		},
		Required: []string{"grade", "comment"},
	}
}
