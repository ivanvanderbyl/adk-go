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

// Package main provides an example ADK server using Anthropic Claude
// with a sub-agent for filesystem exploration and summarization.
//
// Run with:
//
//	ANTHROPIC_API_KEY=your-key go run ./examples/anthropic-rest-filesystem web
//	ANTHROPIC_API_KEY=your-key go run ./examples/anthropic-rest-filesystem console
package main

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/cmd/launcher/web/api"
	"google.golang.org/adk/cmd/launcher/web/webui"
	"google.golang.org/adk/model/anthropic"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/functiontool"
)

// ListDirectoryInput is the input schema for the list_directory tool.
type ListDirectoryInput struct {
	Path      string `json:"path" jsonschema:"the directory path to list"`
	Recursive bool   `json:"recursive,omitempty" jsonschema:"whether to list recursively (default false)"`
	MaxDepth  int    `json:"max_depth,omitempty" jsonschema:"maximum depth for recursive listing (default 3)"`
}

// ListDirectoryOutput is the output schema for the list_directory tool.
type ListDirectoryOutput struct {
	Entries []FileEntry `json:"entries"`
	Error   string      `json:"error,omitempty"`
}

// FileEntry represents a file or directory entry.
type FileEntry struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
}

// listDirectory lists files and directories at the given path.
func listDirectory(_ tool.Context, input ListDirectoryInput) (ListDirectoryOutput, error) {
	if input.Path == "" {
		input.Path = "."
	}
	if input.MaxDepth == 0 {
		input.MaxDepth = 3
	}

	absPath, err := filepath.Abs(input.Path)
	if err != nil {
		return ListDirectoryOutput{Error: fmt.Sprintf("invalid path: %v", err)}, nil
	}

	var entries []FileEntry

	if input.Recursive {
		depth := 0
		err = filepath.WalkDir(absPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			relPath, _ := filepath.Rel(absPath, path)
			currentDepth := strings.Count(relPath, string(os.PathSeparator))
			if currentDepth > input.MaxDepth {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			depth = currentDepth

			info, err := d.Info()
			if err != nil {
				return nil
			}
			entries = append(entries, FileEntry{
				Name:  d.Name(),
				Path:  path,
				IsDir: d.IsDir(),
				Size:  info.Size(),
			})
			return nil
		})
		_ = depth
	} else {
		dirEntries, err := os.ReadDir(absPath)
		if err != nil {
			return ListDirectoryOutput{Error: fmt.Sprintf("failed to read directory: %v", err)}, nil
		}
		for _, entry := range dirEntries {
			info, err := entry.Info()
			if err != nil {
				continue
			}
			entries = append(entries, FileEntry{
				Name:  entry.Name(),
				Path:  filepath.Join(absPath, entry.Name()),
				IsDir: entry.IsDir(),
				Size:  info.Size(),
			})
		}
	}

	return ListDirectoryOutput{Entries: entries}, nil
}

// ReadFileInput is the input schema for the read_file tool.
type ReadFileInput struct {
	Path     string `json:"path" jsonschema:"the file path to read"`
	MaxBytes int    `json:"max_bytes,omitempty" jsonschema:"maximum bytes to read (default 10000)"`
}

// ReadFileOutput is the output schema for the read_file tool.
type ReadFileOutput struct {
	Content   string `json:"content"`
	Size      int64  `json:"size"`
	Truncated bool   `json:"truncated"`
	Error     string `json:"error,omitempty"`
}

// readFile reads the contents of a file.
func readFile(_ tool.Context, input ReadFileInput) (ReadFileOutput, error) {
	if input.MaxBytes == 0 {
		input.MaxBytes = 10000
	}

	info, err := os.Stat(input.Path)
	if err != nil {
		return ReadFileOutput{Error: fmt.Sprintf("failed to stat file: %v", err)}, nil
	}
	if info.IsDir() {
		return ReadFileOutput{Error: "path is a directory, not a file"}, nil
	}

	data, err := os.ReadFile(input.Path)
	if err != nil {
		return ReadFileOutput{Error: fmt.Sprintf("failed to read file: %v", err)}, nil
	}

	content := string(data)
	truncated := false
	if len(data) > input.MaxBytes {
		content = string(data[:input.MaxBytes])
		truncated = true
	}

	return ReadFileOutput{
		Content:   content,
		Size:      info.Size(),
		Truncated: truncated,
	}, nil
}

// FileInfoInput is the input schema for the file_info tool.
type FileInfoInput struct {
	Path string `json:"path" jsonschema:"the file or directory path to get info for"`
}

// FileInfoOutput is the output schema for the file_info tool.
type FileInfoOutput struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Size    int64  `json:"size"`
	IsDir   bool   `json:"is_dir"`
	Mode    string `json:"mode"`
	ModTime string `json:"mod_time"`
	Error   string `json:"error,omitempty"`
}

// fileInfo returns detailed information about a file or directory.
func fileInfo(_ tool.Context, input FileInfoInput) (FileInfoOutput, error) {
	absPath, err := filepath.Abs(input.Path)
	if err != nil {
		return FileInfoOutput{Error: fmt.Sprintf("invalid path: %v", err)}, nil
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return FileInfoOutput{Error: fmt.Sprintf("failed to stat: %v", err)}, nil
	}

	return FileInfoOutput{
		Name:    info.Name(),
		Path:    absPath,
		Size:    info.Size(),
		IsDir:   info.IsDir(),
		Mode:    info.Mode().String(),
		ModTime: info.ModTime().Format("2006-01-02 15:04:05"),
	}, nil
}

func main() {
	ctx := context.Background()

	model, err := anthropic.NewModel(ctx, anthropicsdk.ModelClaude4Sonnet20250514, &anthropic.Config{
		Variant: anthropic.VariantAnthropicAPI,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	listDirTool, err := functiontool.New(functiontool.Config{
		Name:        "list_directory",
		Description: "List files and directories at a given path. Can list recursively with depth control.",
	}, listDirectory)
	if err != nil {
		log.Fatalf("Failed to create list_directory tool: %v", err)
	}

	readFileTool, err := functiontool.New(functiontool.Config{
		Name:        "read_file",
		Description: "Read the contents of a file. Returns the file content as text.",
	}, readFile)
	if err != nil {
		log.Fatalf("Failed to create read_file tool: %v", err)
	}

	fileInfoTool, err := functiontool.New(functiontool.Config{
		Name:        "file_info",
		Description: "Get detailed information about a file or directory including size, permissions, and modification time.",
	}, fileInfo)
	if err != nil {
		log.Fatalf("Failed to create file_info tool: %v", err)
	}

	filesystemAgent, err := llmagent.New(llmagent.Config{
		Name:        "filesystem_explorer",
		Model:       model,
		Description: "Explores the filesystem, lists directories, reads files, and provides information about files and directories.",
		Instruction: `You are a filesystem exploration specialist. Your job is to:
1. Navigate and explore directory structures
2. Read file contents when asked
3. Provide detailed information about files and directories
4. Summarize what you find in a clear and organized manner

When exploring directories:
- Start with a non-recursive listing to get an overview
- Use recursive listing sparingly and with appropriate depth limits
- Group files by type or purpose when summarizing

When reading files:
- Only read files when specifically asked or when needed to understand content
- Summarize file contents rather than dumping raw content
- Be mindful of binary files and skip them

Always provide clear, organized summaries of your findings.`,
		Tools: []tool.Tool{
			listDirTool,
			readFileTool,
			fileInfoTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create filesystem agent: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "filesystem_assistant",
		Model:       model,
		Description: "An assistant that can explore the filesystem and summarize findings.",
		Instruction: `You are a helpful assistant with access to a filesystem exploration agent.

When users ask about files, directories, or want to explore the filesystem:
1. Use the filesystem_explorer agent to perform the exploration
2. Provide clear summaries of what was found
3. Answer follow-up questions about the filesystem content

You can help users:
- List and explore directories
- Read and summarize file contents
- Find specific files or patterns
- Understand project structures
- Summarize codebases or documentation

Always be helpful and provide clear, organized responses.`,
		Tools: []tool.Tool{
			agenttool.New(filesystemAgent, nil),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create root agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:     agent.NewSingleLoader(rootAgent),
		SessionService:  session.InMemoryService(),
		ArtifactService: artifact.InMemoryService(),
	}

	l := web.NewLauncher(api.NewLauncher(), webui.NewLauncher())
	if _, err := l.Parse([]string{"api", "webui"}); err != nil {
		log.Fatalf("Failed to parse args: %v", err)
	}
	if err := l.Run(ctx, config); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
