package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

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

IRサイトに情報がない、またはアクセスできない場合はその旨を記載。
代わりにhttps://irbank.net/からの情報を参考にしてください。
Web検索でもニュースが見つからない場合は「該当ニュースなし」と記載。%s`,
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

// readAloudFiles 指定したディレクトリ内の全ファイルを音声で読み上げる
func readAloudFiles(dirPath string) error {
	// ディレクトリの存在確認
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		return fmt.Errorf("ディレクトリが見つかりません: %s", dirPath)
	}

	// ディレクトリ内のファイル一覧を取得
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return fmt.Errorf("ディレクトリの読み込みに失敗: %w", err)
	}

	txtFiles := []string{}
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".txt") {
			txtFiles = append(txtFiles, file.Name())
		}
	}

	if len(txtFiles) == 0 {
		return fmt.Errorf("読み上げ可能な.txtファイルが見つかりません")
	}

	fmt.Printf("\n=== 音声読み上げ開始 ===\n")
	fmt.Printf("ディレクトリ: %s\n", dirPath)
	fmt.Printf("対象ファイル数: %d\n\n", len(txtFiles))

	for i, filename := range txtFiles {
		filePath := filepath.Join(dirPath, filename)
		
		fmt.Printf("[%d/%d] %s を読み上げ中...\n", i+1, len(txtFiles), filename)
		
		// ファイル内容を読み込む
		content, err := os.ReadFile(filePath)
		if err != nil {
			fmt.Printf("  エラー: ファイルの読み込みに失敗 - %v\n", err)
			continue
		}

		// macOSのsayコマンドで読み上げ
		// -v Kyoko で日本語音声を使用
		cmd := exec.Command("say", "-v", "Kyoko", string(content))
		if err := cmd.Run(); err != nil {
			fmt.Printf("  エラー: 音声読み上げに失敗 - %v\n", err)
			continue
		}

		fmt.Printf("  ✓ 完了\n")
	}

	fmt.Println("\n=== 音声読み上げ完了 ===")
	return nil
}

func main() {
	// コマンドライン引数の定義
	readMode := flag.Bool("read", false, "音声読み上げモード")
	targetDir := flag.String("dir", "", "読み上げ対象のディレクトリパス（例: output/2025-12-06）")
	flag.Parse()

	// 音声読み上げモードの場合
	if *readMode {
		if *targetDir == "" {
			log.Fatal("読み上げモードでは -dir オプションでディレクトリを指定してください\n使用例: go run main.go -read -dir output/2025-12-06")
		}

		if err := readAloudFiles(*targetDir); err != nil {
			log.Fatalf("エラー: %v", err)
		}
		return
	}

	// 以下は既存のニュース取得モード
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

	// 出力ディレクトリを作成（output/YYYY-MM-DD）
	now := time.Now()
	dateStr := now.Format("2006-01-02")
	outputDir := filepath.Join("output", dateStr)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("出力ディレクトリの作成に失敗: %v", err)
	}

	// 各会社のニュースを取得して出力
	fmt.Println("=== 株価関連情報 ===")
	fmt.Printf("出力ディレクトリ: %s\n", outputDir)
	fmt.Println()

	for i, company := range config.Companies {
		fmt.Printf("[%d] %s (%s) を処理中...\n", i+1, company.Name, company.Ticker)

		news, err := getStockNews(apiKey, company)
		if err != nil {
			fmt.Printf("  エラー: %v\n", err)
			continue
		}

		// ファイル名を作成（会社名_証券コード.txt）
		filename := fmt.Sprintf("%s_%s.txt", company.Name, company.Ticker)
		filePath := filepath.Join(outputDir, filename)

		// ファイルに書き込み
		content := fmt.Sprintf("会社名: %s\n証券コード: %s\nIRサイト: %s\n取得日時: %s\n\n%s\n",
			company.Name, company.Ticker, company.IrURL, now.Format("2006-01-02 15:04:05"), news)

		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			fmt.Printf("  ファイル書き込みエラー: %v\n", err)
			continue
		}

		fmt.Printf("  ✓ 完了: %s\n", filePath)
	}

	fmt.Println("\n=== 処理完了 ===")
}
