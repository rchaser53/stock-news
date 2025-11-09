package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Company 会社情報の構造体
type Company struct {
	Name   string `yaml:"name"`
	Ticker string `yaml:"ticker"`
	IrURL  string `yaml:"ir_url"`
}

// Config 設定ファイルの構造体
type Config struct {
	Companies []Company `yaml:"companies"`
}

// loadConfig 設定ファイルを読み込む
func loadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("設定ファイルの読み込みに失敗しました: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("設定ファイルのパースに失敗しました: %w", err)
	}

	return &config, nil
}

// 変更点1: リクエスト型をResponses API向けに
type ResponsesRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
	Tools []struct {
		Type string `json:"type"`
	} `json:"tools,omitempty"`
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`
}

type ResponsesMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ResponsesResponse struct {
	Output []struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		// ツール呼び出しが入る場合もあるが、まずは本文だけ拾う
	} `json:"output"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

func getStockNews(apiKey string, company Company) (string, error) {
	irURLInfo := ""
	if company.IrURL != "" {
		irURLInfo = fmt.Sprintf("\n\n**重要**: 必ず以下のIRサイトも確認してください:\n%s", company.IrURL)
	}

	prompt := fmt.Sprintf(
		`%s（証券コード: %s）について、以下の手順で情報を収集してください:

1. IRサイト（%s）から最新のIR情報（決算、開示資料、プレスリリース）を確認
2. Web検索で直近30日以内の株価関連ニュースを調査

## 出力形式
### IRサイトからの最新情報
- 日付と内容を箇条書き（最大3件）

### Web検索からのニュース
1. **記事タイトル** (YYYY-MM-DD)
   - 要約: [株価への影響を中心に]
   - 出典: [URL]

IRサイトに情報がない、またはアクセスできない場合はその旨を記載し、Web検索結果のみを表示。
両方とも見つからない場合は事業概要を150字で要約。%s`,
		company.Name, company.Ticker, company.IrURL, irURLInfo,
	)

	reqBody := ResponsesRequest{
		Model: "gpt-4o",
		Input: []ResponsesMessage{
			{Role: "user", Content: prompt},
		},
		Tools: []struct {
			Type string `json:"type"`
		}{
			{Type: "web_search"},
		},
		MaxOutputTokens: 1500,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("JSON化に失敗: %w", err)
	}

	// 変更点2: エンドポイントを /v1/responses に
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/responses", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("リクエスト作成失敗: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("APIリクエスト失敗: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("APIエラー (status %d): %s", resp.StatusCode, string(body))
	}

	var r ResponsesResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return "", fmt.Errorf("パース失敗: %w", err)
	}
	if r.Error != nil {
		return "", fmt.Errorf("OpenAI APIエラー: %s", r.Error.Message)
	}

	// Output配列から最後の要素（テキスト出力）を取得
	if len(r.Output) == 0 {
		return "", fmt.Errorf("応答が空です")
	}

	// 最後の出力要素を取得（通常は最後にテキストが入る）
	lastOutput := r.Output[len(r.Output)-1]
	if len(lastOutput.Content) == 0 {
		return "", fmt.Errorf("応答内容が空です")
	}

	// Type が "output_text" の要素を探す
	for _, content := range lastOutput.Content {
		if content.Type == "output_text" && content.Text != "" {
			return content.Text, nil
		}
	}

	return "", fmt.Errorf("テキスト応答が見つかりません")
}

func main() {
	// OpenAI APIキーを環境変数から取得
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("環境変数 OPENAI_API_KEY が設定されていません")
	}

	// 設定ファイルを読み込む
	config, err := loadConfig("config.yaml")
	if err != nil {
		log.Fatalf("設定ファイルの読み込みエラー: %v", err)
	}

	if len(config.Companies) == 0 {
		log.Fatal("設定ファイルに会社が登録されていません")
	}

	// 各会社のニュースを取得して出力
	fmt.Println("=== 株価関連情報 ===")
	fmt.Println()

	for i, company := range config.Companies {
		fmt.Printf("[%d] %s (%s)\n", i+1, company.Name, company.Ticker)
		fmt.Println(strings.Repeat("-", 60))

		news, err := getStockNews(apiKey, company)
		if err != nil {
			fmt.Printf("エラー: %v\n", err)
		} else {
			fmt.Println(news)
		}

		fmt.Println()
	}
}
