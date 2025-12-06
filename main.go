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
	"sort"
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

// getLatestOutputDir 最新の出力ディレクトリを取得（現在日を除く）
func getLatestOutputDir(excludeDate string) (string, error) {
	outputBase := "output"
	entries, err := os.ReadDir(outputBase)
	if err != nil {
		return "", fmt.Errorf("outputディレクトリの読み込みに失敗: %w", err)
	}

	var dirs []string
	for _, entry := range entries {
		if entry.IsDir() && entry.Name() != excludeDate {
			dirs = append(dirs, entry.Name())
		}
	}

	if len(dirs) == 0 {
		return "", fmt.Errorf("比較対象のディレクトリがありません")
	}

	// 日付順にソート（降順）
	sort.Slice(dirs, func(i, j int) bool {
		return dirs[i] > dirs[j]
	})

	return filepath.Join(outputBase, dirs[0]), nil
}

// extractNewsContent ファイルからニュース本文部分のみを抽出
func extractNewsContent(content string) string {
	lines := strings.Split(content, "\n")
	var newsLines []string
	skipHeader := true

	for _, line := range lines {
		// ヘッダー部分をスキップ（空行が出るまで）
		if skipHeader {
			if strings.TrimSpace(line) == "" && len(newsLines) == 0 {
				skipHeader = false
			}
			continue
		}
		newsLines = append(newsLines, line)
	}

	return strings.TrimSpace(strings.Join(newsLines, "\n"))
}

// summarizeDiff 差分内容を要約する
func summarizeDiff(apiKey, companyName, oldContent, newContent string) (string, error) {
	prompt := fmt.Sprintf(
		`以下は%sに関する株価情報の旧版と新版です。
変更点を簡潔に要約してください（200文字以内）。

## 旧版の内容
%s

## 新版の内容
%s

## 要約の形式
変更点を箇条書きで3点以内にまとめてください。株価に影響する重要な情報を優先してください。`,
		companyName, oldContent, newContent,
	)

	reqBody := ResponsesRequest{
		Model: "gpt-4o-mini",
		Input: []ResponsesMessage{
			{Role: "user", Content: prompt},
		},
		MaxOutputTokens: 300,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("JSON化に失敗: %w", err)
	}

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

	if len(r.Output) == 0 {
		return "", fmt.Errorf("応答が空です")
	}

	lastOutput := r.Output[len(r.Output)-1]
	if len(lastOutput.Content) == 0 {
		return "", fmt.Errorf("応答内容が空です")
	}

	for _, content := range lastOutput.Content {
		if content.Type == "output_text" && content.Text != "" {
			return content.Text, nil
		}
	}

	return "", fmt.Errorf("テキスト応答が見つかりません")
}

// DiffInfo 差分情報の構造体
type DiffInfo struct {
	FilePath    string
	FileName    string
	CompanyName string
	OldContent  string
	NewContent  string
}

// compareAndReadDiffs 新旧ファイルを比較し、差分があれば要約して読み上げる
func compareAndReadDiffs(apiKey, newDir, oldDir string) error {
	fmt.Printf("\n=== 差分チェック ===\n")
	fmt.Printf("新規データ: %s\n", newDir)
	fmt.Printf("比較対象: %s\n\n", oldDir)

	newFiles, err := os.ReadDir(newDir)
	if err != nil {
		return fmt.Errorf("新規ディレクトリの読み込みに失敗: %w", err)
	}

	hasChanges := false
	var diffs []DiffInfo

	for _, file := range newFiles {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".txt") {
			continue
		}

		newFilePath := filepath.Join(newDir, file.Name())
		oldFilePath := filepath.Join(oldDir, file.Name())

		// 新規ファイルを読み込む
		newContent, err := os.ReadFile(newFilePath)
		if err != nil {
			fmt.Printf("  %s: 読み込みエラー - %v\n", file.Name(), err)
			continue
		}

		// ファイル名から会社名を抽出（例: スポーツフィールド_7080.txt -> スポーツフィールド）
		companyName := strings.Split(file.Name(), "_")[0]

		// 旧ファイルが存在するか確認
		oldContent, err := os.ReadFile(oldFilePath)
		if err != nil {
			// 旧ファイルが存在しない場合は新規として扱う
			fmt.Printf("  %s: 新規ファイル（差分あり）\n", file.Name())
			hasChanges = true
			diffs = append(diffs, DiffInfo{
				FilePath:    newFilePath,
				FileName:    file.Name(),
				CompanyName: companyName,
				OldContent:  "",
				NewContent:  extractNewsContent(string(newContent)),
			})
			continue
		}

		// ニュース本文部分のみを比較
		newNews := extractNewsContent(string(newContent))
		oldNews := extractNewsContent(string(oldContent))

		if newNews != oldNews {
			fmt.Printf("  %s: 差分検出\n", file.Name())
			hasChanges = true
			diffs = append(diffs, DiffInfo{
				FilePath:    newFilePath,
				FileName:    file.Name(),
				CompanyName: companyName,
				OldContent:  oldNews,
				NewContent:  newNews,
			})
		} else {
			fmt.Printf("  %s: 変更なし\n", file.Name())
		}
	}

	if !hasChanges {
		fmt.Println("\n変更のあるファイルはありませんでした。")
		return nil
	}

	fmt.Printf("\n=== 差分を要約して読み上げます（%d件） ===\n\n", len(diffs))

	for i, diff := range diffs {
		fmt.Printf("[%d/%d] %s の差分を処理中...\n", i+1, len(diffs), diff.FileName)

		var summaryText string
		if diff.OldContent == "" {
			// 新規ファイルの場合
			summaryText = fmt.Sprintf("%sの新規情報です。%s", diff.CompanyName, diff.NewContent)
			fmt.Printf("  新規ファイル: 全文を読み上げます\n")
		} else {
			// 差分がある場合は要約を生成
			fmt.Printf("  差分を要約中...\n")
			summary, err := summarizeDiff(apiKey, diff.CompanyName, diff.OldContent, diff.NewContent)
			if err != nil {
				fmt.Printf("  警告: 要約の生成に失敗 - %v\n", err)
				fmt.Printf("  元の内容を読み上げます\n")
				summaryText = fmt.Sprintf("%sに関する更新情報です。%s", diff.CompanyName, diff.NewContent)
			} else {
				summaryText = fmt.Sprintf("%sに関する変更点です。%s", diff.CompanyName, summary)
				fmt.Printf("  要約完了\n")
			}
		}

		// 要約を音声で読み上げ
		fmt.Printf("  読み上げ中...\n")
		cmd := exec.Command("say", "-v", "Kyoko", summaryText)
		if err := cmd.Run(); err != nil {
			fmt.Printf("  エラー: 音声読み上げに失敗 - %v\n", err)
			continue
		}

		fmt.Printf("  ✓ 完了\n\n")
	}

	fmt.Println("=== 差分読み上げ完了 ===")
	return nil
}

func main() {
	// コマンドライン引数の定義
	readMode := flag.Bool("read", false, "音声読み上げモード")
	targetDir := flag.String("dir", "", "読み上げ対象のディレクトリパス（例: output/2025-12-06）")
	autoRead := flag.Bool("auto-read", false, "情報取得後に自動で差分を読み上げる")
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

	// 自動読み上げモードの場合、差分をチェックして読み上げ
	if *autoRead {
		fmt.Println()
		latestDir, err := getLatestOutputDir(dateStr)
		if err != nil {
			fmt.Printf("比較対象のディレクトリが見つかりません: %v\n", err)
			fmt.Println("（初回実行のため、比較をスキップします）")
		} else {
			if err := compareAndReadDiffs(apiKey, outputDir, latestDir); err != nil {
				log.Printf("差分チェックでエラーが発生: %v", err)
			}
		}
	}
}
