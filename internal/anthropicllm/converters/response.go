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

package converters

import (
	"encoding/base64"
	"encoding/json"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// MessageToLLMResponse converts an Anthropic Message to a model.LLMResponse.
func MessageToLLMResponse(msg *anthropic.Message) *model.LLMResponse {
	if msg == nil {
		return &model.LLMResponse{
			ErrorCode:    "UNKNOWN_ERROR",
			ErrorMessage: "nil message received",
		}
	}

	content := &genai.Content{
		Role:  "model",
		Parts: make([]*genai.Part, 0, len(msg.Content)),
	}

	for _, block := range msg.Content {
		part := ContentBlockToGenaiPart(block)
		if part != nil {
			content.Parts = append(content.Parts, part)
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: UsageToMetadata(msg.Usage),
		FinishReason:  StopReasonToFinishReason(msg.StopReason),
	}
}

// ContentBlockToGenaiPart converts an Anthropic ContentBlockUnion to a genai.Part.
func ContentBlockToGenaiPart(block anthropic.ContentBlockUnion) *genai.Part {
	switch variant := block.AsAny().(type) {
	case anthropic.TextBlock:
		return &genai.Part{Text: variant.Text}

	case anthropic.ThinkingBlock:
		// Map thinking blocks to genai.Part with Thought=true
		signature, _ := base64.StdEncoding.DecodeString(variant.Signature)
		return &genai.Part{
			Text:             variant.Thinking,
			Thought:          true,
			ThoughtSignature: signature,
		}

	case anthropic.RedactedThinkingBlock:
		// Redacted thinking - we can't see the content but preserve the marker
		return &genai.Part{
			Text:    "[thinking redacted]",
			Thought: true,
		}

	case anthropic.ToolUseBlock:
		// Convert to FunctionCall
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is json.RawMessage, unmarshal it
			_ = json.Unmarshal(variant.Input, &args)
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: variant.Name,
				Args: args,
			},
		}

	case anthropic.ServerToolUseBlock:
		// Server-side tool use (web search, etc.)
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is an any type, convert through JSON
			if inputBytes, err := json.Marshal(variant.Input); err == nil {
				_ = json.Unmarshal(inputBytes, &args)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: string(variant.Name),
				Args: args,
			},
		}

	default:
		// Unknown block type - skip
		return nil
	}
}

// UsageToMetadata converts Anthropic Usage to genai UsageMetadata.
func UsageToMetadata(usage anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.InputTokens + usage.OutputTokens),
	}
}

// StopReasonToFinishReason maps Anthropic StopReason to genai FinishReason.
func StopReasonToFinishReason(sr anthropic.StopReason) genai.FinishReason {
	switch sr {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonUnspecified
	}
}

// StreamDeltaToPartialResponse converts a streaming content block delta to a partial LLMResponse.
// Used for streaming text updates.
func StreamDeltaToPartialResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: text},
			},
		},
		Partial: true,
	}
}

// StreamThinkingDeltaToPartialResponse converts a streaming thinking delta to a partial LLMResponse.
func StreamThinkingDeltaToPartialResponse(thinking string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:    thinking,
					Thought: true,
				},
			},
		},
		Partial: true,
	}
}
