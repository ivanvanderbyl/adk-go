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

// Package main provides an example research agent that generates comprehensive,
// cited research reports using iterative evaluation and refinement.
//
// Run with:
//
//	go run ./examples/research
//
// Environment variables:
//   - ANTHROPIC_API_KEY: API key for Anthropic Claude models
//   - GOOGLE_GENAI_API_KEY: API key for Gemini (used for Google Search)
//   - MAX_SEARCH_ITERATIONS: Maximum refinement iterations (default: 25)
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"os"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/anthropic"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	ctx := context.Background()
	cfg := NewConfig()

	claudeModel, err := anthropic.NewModel(ctx, anthropicsdk.ModelClaudeHaiku4_5, nil)
	if err != nil {
		log.Fatalf("Failed to create Anthropic model: %v", err)
	}

	geminiModel, err := gemini.NewModel(ctx, "gemini-2.5-flash", nil)
	if err != nil {
		log.Fatalf("Failed to create Gemini model: %v", err)
	}

	rootAgent, err := buildResearchAgent(claudeModel, geminiModel, cfg)
	if err != nil {
		log.Fatalf("Failed to build research agent: %v", err)
	}

	launcherConfig := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, launcherConfig, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func buildResearchAgent(llm model.LLM, searchModel model.LLM, cfg *Config) (agent.Agent, error) {
	currentDate := time.Now().Format("2006-01-02")

	// Create a Gemini-powered Google Search sub-agent
	googleSearchAgent, err := llmagent.New(llmagent.Config{
		Name:        "web_search",
		Model:       searchModel,
		Description: "Searches the web using Google Search and returns relevant information. Use this to find current information, facts, and research data.",
		Instruction: `You are a web search assistant. When given a search query:
1. Use the google_search tool to find relevant information
2. Summarize the key findings from the search results
3. Include relevant URLs and sources in your response
4. Be concise but comprehensive in your summary

Always cite your sources and provide factual, accurate information from the search results.`,
		Tools: []tool.Tool{geminitool.GoogleSearch{}},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create google_search_agent: %w", err)
	}

	// Wrap the search agent as a tool for Claude to use
	searchTool := agenttool.New(googleSearchAgent, nil)

	planGenerator, err := llmagent.New(llmagent.Config{
		Name:        "plan_generator",
		Model:       llm,
		Description: "Generates or refines the existing 5 line action-oriented research plan.",
		Instruction: fmt.Sprintf(`
You are a research strategist. Your job is to create a high-level RESEARCH PLAN, not a summary. If there is already a RESEARCH PLAN in the session state,
improve upon it based on the user feedback.

RESEARCH PLAN(SO FAR):
{research_plan?}

**GENERAL INSTRUCTION: CLASSIFY TASK TYPES**
Your plan must clearly classify each goal for downstream execution. Each bullet point should start with a task type prefix:
- **[RESEARCH]**: For goals that primarily involve information gathering, investigation, analysis, or data collection.
- **[DELIVERABLE]**: For goals that involve synthesizing collected information, creating structured outputs (e.g., tables, charts, summaries, reports), or compiling final output artifacts (these are executed AFTER research tasks).

**INITIAL RULE: Your initial output MUST start with a bulleted list of 5 action-oriented research goals or key questions, followed by any *inherently implied* deliverables.**
- All initial 5 goals will be classified as [RESEARCH] tasks.
- A good goal for [RESEARCH] starts with a verb like "Analyze," "Identify," "Investigate."
- A bad output is a statement of fact like "The event was in April 2024."
- **Proactive Implied Deliverables (Initial):** If any of your initial 5 [RESEARCH] goals inherently imply a standard output or deliverable (e.g., a comparative analysis suggesting a comparison table, or a comprehensive review suggesting a summary document), you MUST add these as additional, distinct goals immediately after the initial 5. Phrase these as *synthesis or output creation actions* (e.g., "Create a summary," "Develop a comparison," "Compile a report") and prefix them with [DELIVERABLE][IMPLIED].

**REFINEMENT RULE**:
- **Integrate Feedback & Mark Changes:** When incorporating user feedback, make targeted modifications to existing bullet points. Add [MODIFIED] to the existing task type and status prefix (e.g., [RESEARCH][MODIFIED]). If the feedback introduces new goals:
    - If it's an information gathering task, prefix it with [RESEARCH][NEW].
    - If it's a synthesis or output creation task, prefix it with [DELIVERABLE][NEW].
- **Proactive Implied Deliverables (Refinement):** Beyond explicit user feedback, if the nature of an existing [RESEARCH] goal (e.g., requiring a structured comparison, deep dive analysis, or broad synthesis) or a [DELIVERABLE] goal inherently implies an additional, standard output or synthesis step (e.g., a detailed report following a summary, or a visual representation of complex data), proactively add this as a new goal. Phrase these as *synthesis or output creation actions* and prefix them with [DELIVERABLE][IMPLIED].
- **Maintain Order:** Strictly maintain the original sequential order of existing bullet points. New bullets, whether [NEW] or [IMPLIED], should generally be appended to the list, unless the user explicitly instructs a specific insertion point.
- **Flexible Length:** The refined plan is no longer constrained by the initial 5-bullet limit and may comprise more goals as needed to fully address the feedback and implied deliverables.

Current date: %s
`, currentDate),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create plan_generator: %w", err)
	}

	sectionPlanner, err := llmagent.New(llmagent.Config{
		Name:        "section_planner",
		Model:       llm,
		Description: "Breaks down the research plan into a structured markdown outline of report sections.",
		Instruction: `
You are an expert report architect. Using the research topic and the plan from the 'research_plan' state key, design a logical structure for the final report.
Note: Ignore all the tag names ([MODIFIED], [NEW], [RESEARCH], [DELIVERABLE]) in the research plan.
Your task is to create a markdown outline with 4-6 distinct sections that cover the topic comprehensively without overlap.
You can use any markdown format you prefer, but here's a suggested structure:
# Section Name
A brief overview of what this section covers
Feel free to add subsections or bullet points if needed to better organize the content.
Make sure your outline is clear and easy to follow.
Do not include a "References" or "Sources" section in your outline. Citations will be handled in-line.
`,
		OutputKey: "report_sections",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create section_planner: %w", err)
	}

	sectionResearcher, err := llmagent.New(llmagent.Config{
		Name:        "section_researcher",
		Model:       llm,
		Description: "Performs the crucial first pass of web research.",
		Instruction: `
You are a highly capable and diligent research and synthesis agent. Your comprehensive task is to execute a provided research plan with **absolute fidelity**, using the web_search tool to gather information and then synthesizing it.

You will be provided with a sequential list of research plan goals, stored in the research_plan state key. Each goal will be clearly prefixed with its primary task type: [RESEARCH] or [DELIVERABLE].

Your execution process must strictly adhere to these two distinct and sequential phases:

---

**Phase 1: Information Gathering ([RESEARCH] Tasks)**

*   **Execution Directive:** You **MUST** systematically process every goal prefixed with [RESEARCH] before proceeding to Phase 2.
*   For each [RESEARCH] goal:
    *   **Query Generation:** Formulate 3-5 targeted search queries to address the [RESEARCH] goal comprehensively.
    *   **Web Search:** Use the web_search tool to execute your queries and gather current, accurate information.
    *   **Summarization:** Synthesize the search results into a detailed, coherent summary that directly addresses the objective of the [RESEARCH] goal.
    *   **Internal Storage:** Store this summary, clearly tagged or indexed by its corresponding [RESEARCH] goal, for later and exclusive use in Phase 2.

---

**Phase 2: Synthesis and Output Creation ([DELIVERABLE] Tasks)**

*   **Execution Prerequisite:** This phase **MUST ONLY COMMENCE** once **ALL** [RESEARCH] goals from Phase 1 have been fully completed and their summaries are internally stored.
*   **Execution Directive:** You **MUST** systematically process **every** goal prefixed with [DELIVERABLE]. For each [DELIVERABLE] goal, your directive is to **PRODUCE** the artifact as explicitly described.
*   For each [DELIVERABLE] goal:
    *   **Instruction Interpretation:** You will interpret the goal's text (following the [DELIVERABLE] tag) as a **direct and non-negotiable instruction** to generate a specific output artifact.
        *   *If the instruction details a table (e.g., "Create a Detailed Comparison Table in Markdown format"), your output for this step **MUST** be a properly formatted Markdown table utilizing columns and rows as implied by the instruction and the prepared data.*
        *   *If the instruction states to prepare a summary, report, or any other structured output, your output for this step **MUST** be that precise artifact.*
    *   **Data Consolidation:** Access and utilize **ONLY** the summaries generated during Phase 1 ([RESEARCH] tasks) to fulfill the requirements of the current [DELIVERABLE] goal.
    *   **Output Generation:** Based on the specific instruction of the [DELIVERABLE] goal:
        *   Carefully extract, organize, and synthesize the relevant information from your previously gathered summaries.
        *   Must always produce the specified output artifact (e.g., a concise summary, a structured comparison table, a comprehensive report, a visual representation, etc.) with accuracy and completeness.
    *   **Output Accumulation:** Maintain and accumulate **all** the generated [DELIVERABLE] artifacts. These are your final outputs.

---

**Final Output:** Your final output will comprise the complete set of processed summaries from [RESEARCH] tasks AND all the generated artifacts from [DELIVERABLE] tasks, presented clearly and distinctly.
`,
		Tools:     []tool.Tool{searchTool},
		OutputKey: "section_research_findings",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create section_researcher: %w", err)
	}

	researchEvaluator, err := llmagent.New(llmagent.Config{
		Name:        "research_evaluator",
		Model:       llm,
		Description: "Critically evaluates research and generates follow-up queries.",
		Instruction: fmt.Sprintf(`
You are a meticulous quality assurance analyst evaluating the research findings in 'section_research_findings'.

**CRITICAL RULES:**
1. Assume the given research topic is correct. Do not question or try to verify the subject itself.
2. Your ONLY job is to assess the quality, depth, and completeness of the research provided *for that topic*.
3. Focus on evaluating: Comprehensiveness of coverage, logical flow and organization, use of credible sources, depth of analysis, and clarity of explanations.
4. Do NOT fact-check or question the fundamental premise or timeline of the topic.
5. If suggesting follow-up queries, they should dive deeper into the existing topic, not question its validity.

Be very critical about the QUALITY of research. If you find significant gaps in depth or coverage, assign a grade of "fail",
write a detailed comment about what's missing, and generate 5-7 specific follow-up queries to fill those gaps.
If the research thoroughly covers the topic, grade "pass".

Current date: %s
Your response must be a single, raw JSON object validating against the 'Feedback' schema.
`, currentDate),
		OutputSchema:             FeedbackSchema(),
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
		OutputKey:                "research_evaluation",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create research_evaluator: %w", err)
	}

	escalationChecker, err := newEscalationChecker("escalation_checker")
	if err != nil {
		return nil, fmt.Errorf("failed to create escalation_checker: %w", err)
	}

	enhancedSearchExecutor, err := llmagent.New(llmagent.Config{
		Name:        "enhanced_search_executor",
		Model:       llm,
		Description: "Executes follow-up web searches and integrates new findings.",
		Instruction: `
You are a specialist researcher executing a refinement pass.
You have been activated because the previous research was graded as 'fail'.

1.  Review the 'research_evaluation' state key to understand the feedback and required fixes.
2.  Use the web_search tool to execute EVERY query listed in 'follow_up_queries'.
3.  Synthesize the new findings and COMBINE them with the existing information in 'section_research_findings'.
4.  Your output MUST be the new, complete, and improved set of research findings.
`,
		Tools:     []tool.Tool{searchTool},
		OutputKey: "section_research_findings",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create enhanced_search_executor: %w", err)
	}

	iterativeRefinementLoop, err := loopagent.New(loopagent.Config{
		MaxIterations: cfg.MaxSearchIterations,
		AgentConfig: agent.Config{
			Name:        "iterative_refinement_loop",
			Description: "Iteratively refines research until quality threshold is met",
			SubAgents:   []agent.Agent{researchEvaluator, escalationChecker, enhancedSearchExecutor},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create iterative_refinement_loop: %w", err)
	}

	reportComposer, err := llmagent.New(llmagent.Config{
		Name:            "report_composer",
		Model:           llm,
		IncludeContents: llmagent.IncludeContentsNone,
		Description:     "Transforms research data and a markdown outline into a final report.",
		Instruction: `
Transform the provided data into a polished, professional research report.

---
### INPUT DATA
*   Research Plan: {research_plan}
*   Research Findings: {section_research_findings}
*   Report Structure: {report_sections}

---
### Final Instructions
Generate a comprehensive report based on the research findings.
The final report must strictly follow the structure provided in the **Report Structure** markdown outline.
Ensure the report is well-organized, clearly written, and thoroughly addresses all research goals.
`,
		OutputKey: "final_report",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create report_composer: %w", err)
	}

	researchPipeline, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "research_pipeline",
			Description: "Executes a pre-approved research plan. It performs iterative research, evaluation, and composes a final, cited report.",
			SubAgents: []agent.Agent{
				sectionPlanner,
				sectionResearcher,
				iterativeRefinementLoop,
				reportComposer,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create research_pipeline: %w", err)
	}

	planGeneratorTool := agenttool.New(planGenerator, nil)

	interactivePlannerAgent, err := llmagent.New(llmagent.Config{
		Name:        "interactive_planner_agent",
		Model:       llm,
		Description: "The primary research assistant. It collaborates with the user to create a research plan, and then executes it upon approval.",
		Instruction: fmt.Sprintf(`
You are a research planning assistant. Your primary function is to convert ANY user request into a research plan.

**CRITICAL RULE: Never answer a question directly or refuse a request.** Your one and only first step is to use the plan_generator tool to propose a research plan for the user's topic.
If the user asks a question, you MUST immediately call plan_generator to create a plan to answer the question.

Your workflow is:
1.  **Plan:** Use plan_generator to create a draft plan and present it to the user.
2.  **Refine:** Incorporate user feedback until the plan is approved.
3.  **Execute:** Once the user gives EXPLICIT approval (e.g., "looks good, run it"), you MUST delegate the task to the research_pipeline agent, passing the approved plan.

Current date: %s
Do not perform any research yourself. Your job is to Plan, Refine, and Delegate.
`, currentDate),
		SubAgents: []agent.Agent{researchPipeline},
		Tools:     []tool.Tool{planGeneratorTool},
		OutputKey: "research_plan",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create interactive_planner_agent: %w", err)
	}

	return interactivePlannerAgent, nil
}

// newEscalationChecker creates a custom agent that checks research evaluation
// and escalates to stop the loop if the grade is 'pass'.
func newEscalationChecker(name string) (agent.Agent, error) {
	checker := &escalationChecker{name: name}
	return agent.New(agent.Config{
		Name:        name,
		Description: "Checks research evaluation and escalates to stop the loop if grade is 'pass'.",
		Run:         checker.Run,
	})
}

type escalationChecker struct {
	name string
}

func (e *escalationChecker) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		state := ctx.Session().State()

		evaluationVal, err := state.Get("research_evaluation")
		if err != nil {
			log.Printf("[%s] Research evaluation not found in state. Loop will continue for more research.", e.name)
			yield(session.NewEvent(ctx.InvocationID()), nil)
			return
		}

		shouldEscalate := false

		switch evaluation := evaluationVal.(type) {
		case string:
			shouldEscalate = checkGradeInString(evaluation)
		case map[string]any:
			if grade, ok := evaluation["grade"].(string); ok {
				shouldEscalate = (grade == "pass")
			}
		case *Feedback:
			shouldEscalate = (evaluation.Grade == "pass")
		case Feedback:
			shouldEscalate = (evaluation.Grade == "pass")
		default:
			log.Printf("[%s] Unknown evaluation type: %T. Loop will continue.", e.name, evaluationVal)
		}

		if shouldEscalate {
			log.Printf("[%s] Research evaluation PASSED. Escalating to stop refinement loop.", e.name)
			event := session.NewEvent(ctx.InvocationID())
			event.Author = e.name
			event.Actions.Escalate = true
			yield(event, nil)
			return
		}

		log.Printf("[%s] Research evaluation FAILED. Loop will continue for refinement.", e.name)
		event := session.NewEvent(ctx.InvocationID())
		event.Author = e.name
		yield(event, nil)
	}
}

func checkGradeInString(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, `"grade":"pass"`) ||
		strings.Contains(s, `"grade": "pass"`) ||
		strings.Contains(s, `"grade" : "pass"`)
}
