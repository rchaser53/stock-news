package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
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

// Gemini API (generateContent)
// Ref: https://ai.google.dev/api/rest/v1beta/models/generateContent

type GeminiGenerateContentRequest struct {
	Contents         []GeminiContent         `json:"contents"`
	Tools            []GeminiTool            `json:"tools,omitempty"`
	GenerationConfig *GeminiGenerationConfig `json:"generationConfig,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiPart struct {
	Text string `json:"text,omitempty"`
}

type GeminiTool struct {
	// Generative Language APIでは camelCase が期待されることがある
	GoogleSearch map[string]any `json:"googleSearch,omitempty"`
}

type GeminiGenerationConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
}

type GeminiGenerateContentResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

type GeminiListModelsResponse struct {
	Models []struct {
		Name                       string   `json:"name"`
		SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
	} `json:"models"`
	NextPageToken string `json:"nextPageToken"`
	Error         *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

var lastGeminiAPIVersionUsed string

func recordGeminiAPIVersionUsed(apiVersion string) {
	lastGeminiAPIVersionUsed = apiVersion
}

func getLastGeminiAPIVersionUsed() string {
	return lastGeminiAPIVersionUsed
}

func geminiAPIVersionFromEnv() string {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GEMINI_API_VERSION")))
	// ユーザー指定が無い場合は v3 を優先
	if v == "" {
		return "v3"
	}
	if v == "3" {
		return "v3"
	}
	if v != "v1" && v != "v1beta" && v != "v3" {
		return "v3"
	}
	return v
}

func effectiveGeminiAPIVersion(requested string) string {
	// 2025-12 時点で Generative Language API (generativelanguage.googleapis.com) のRESTは v1/v1beta が主。
	// ユーザーが "v3" を指定した場合でも、実際には動作するバージョンにマップして動かす。
	if requested == "v3" {
		return "v1beta"
	}
	return requested
}

func isAPIVersionNotSupportedError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	// 代表的な失敗パターンを雑に拾う（404/unknown endpoint など）
	return strings.Contains(msg, "404") ||
		strings.Contains(msg, "not found") ||
		strings.Contains(msg, "unknown") ||
		strings.Contains(msg, "unimplemented") ||
		strings.Contains(msg, "invalid argument") && strings.Contains(msg, "version")
}

func normalizeGeminiModelName(model string) string {
	m := strings.TrimSpace(model)
	m = strings.TrimPrefix(m, "models/")
	return m
}

func containsString(list []string, needle string) bool {
	for _, s := range list {
		if s == needle {
			return true
		}
	}
	return false
}

func listGeminiModels(apiKey, apiVersion string) ([]string, error) {
	apiVersion = effectiveGeminiAPIVersion(apiVersion)
	endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/%s/models?key=%s", apiVersion, url.QueryEscape(apiKey))
	req, err := http.NewRequest("GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("リクエスト作成失敗: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("APIリクエスト失敗: %w", err)
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, fmt.Errorf("API応答の読み込み失敗 (status %d): %w", resp.StatusCode, readErr)
	}
	if len(body) == 0 {
		return nil, fmt.Errorf("Gemini ListModels応答が空です (status %d)", resp.StatusCode)
	}

	var r GeminiListModelsResponse
	if err := json.Unmarshal(body, &r); err != nil {
		// JSON以外（HTMLなど）もあり得るので、状況が分かるように status と body を出す
		return nil, fmt.Errorf("Gemini ListModels JSONパース失敗 (status %d, body=%q): %w", resp.StatusCode, string(body), err)
	}
	if resp.StatusCode != http.StatusOK {
		if r.Error != nil {
			return nil, fmt.Errorf("Gemini ListModelsエラー (status %d): %s", resp.StatusCode, r.Error.Message)
		}
		return nil, fmt.Errorf("Gemini ListModelsエラー (status %d): %s", resp.StatusCode, string(body))
	}
	if r.Error != nil {
		return nil, fmt.Errorf("Gemini ListModelsエラー: %s", r.Error.Message)
	}

	var models []string
	for _, m := range r.Models {
		if !containsString(m.SupportedGenerationMethods, "generateContent") {
			continue
		}
		models = append(models, normalizeGeminiModelName(m.Name))
	}
	return models, nil
}

func pickPreferredGeminiModel(models []string) string {
	// 優先順: flash系 → pro系 → その他
	for _, m := range models {
		if strings.Contains(m, "flash") {
			return m
		}
	}
	for _, m := range models {
		if strings.Contains(m, "pro") {
			return m
		}
	}
	if len(models) > 0 {
		return models[0]
	}
	return ""
}

func isModelNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not found") || strings.Contains(msg, "not_supported") || strings.Contains(msg, "not supported")
}

func geminiModelFromEnv(envKey, fallback string) string {
	model := strings.TrimSpace(os.Getenv(envKey))
	if model == "" {
		return fallback
	}
	return model
}

func geminiBoolFromEnv(envKey string, defaultValue bool) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(envKey)))
	if v == "" {
		return defaultValue
	}
	switch v {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	default:
		return defaultValue
	}
}

func callGeminiGenerateContent(apiKey, model, prompt string, maxOutputTokens int, enableGoogleSearch bool) (string, error) {
	if strings.TrimSpace(apiKey) == "" {
		return "", fmt.Errorf("GEMINI_API_KEY が空です")
	}
	model = normalizeGeminiModelName(model)
	if model == "" {
		return "", fmt.Errorf("Gemini model が空です")
	}

	text, err := callGeminiGenerateContentOnce(apiKey, model, prompt, maxOutputTokens, enableGoogleSearch)
	if err == nil {
		return text, nil
	}

	// APIバージョン未対応の環境向けフォールバック
	if isAPIVersionNotSupportedError(err) {
		for _, v := range []string{"v1", "v1beta"} {
			fallbackText, fallbackErr := callGeminiGenerateContentOnceWithVersion(v, apiKey, model, prompt, maxOutputTokens, enableGoogleSearch)
			if fallbackErr == nil {
				return fallbackText, nil
			}
		}
	}

	// モデル名の揺れ対策: `gemini-1.5-flash` が無い環境では `-latest` が提供されている場合がある
	if isModelNotFoundError(err) {
		if !strings.HasSuffix(model, "-latest") {
			fallbackModel := model + "-latest"
			fallbackText, fallbackErr := callGeminiGenerateContentOnce(apiKey, fallbackModel, prompt, maxOutputTokens, enableGoogleSearch)
			if fallbackErr == nil {
				return fallbackText, nil
			}
		}

		// それでもダメなら ListModels で利用可能モデルを探して自動選択
		apiVersion := geminiAPIVersionFromEnv()
		models, listErr := listGeminiModels(apiKey, apiVersion)
		if listErr != nil {
			// v3/v1 でダメなら v1beta も試す
			models, _ = listGeminiModels(apiKey, "v1beta")
		}
		chosen := pickPreferredGeminiModel(models)
		if chosen != "" && chosen != model {
			autoText, autoErr := callGeminiGenerateContentOnce(apiKey, chosen, prompt, maxOutputTokens, enableGoogleSearch)
			if autoErr == nil {
				return autoText, nil
			}
		}
	}

	return "", err
}

func callGeminiGenerateContentOnce(apiKey, model, prompt string, maxOutputTokens int, enableGoogleSearch bool) (string, error) {
	apiVersion := geminiAPIVersionFromEnv()
	return callGeminiGenerateContentOnceWithVersion(apiVersion, apiKey, model, prompt, maxOutputTokens, enableGoogleSearch)
}

func callGeminiGenerateContentOnceWithVersion(apiVersion, apiKey, model, prompt string, maxOutputTokens int, enableGoogleSearch bool) (string, error) {
	apiVersion = effectiveGeminiAPIVersion(apiVersion)

	reqBody := GeminiGenerateContentRequest{
		Contents: []GeminiContent{
			{
				Role:  "user",
				Parts: []GeminiPart{{Text: prompt}},
			},
		},
		GenerationConfig: &GeminiGenerationConfig{MaxOutputTokens: maxOutputTokens},
	}
	// tools は v1 では拒否されることがあるため、v1 以外のときだけ付与
	if enableGoogleSearch && apiVersion != "v1" {
		reqBody.Tools = []GeminiTool{{GoogleSearch: map[string]any{}}}
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("JSON化に失敗: %w", err)
	}

	endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/%s/models/%s:generateContent?key=%s", apiVersion, url.PathEscape(model), url.QueryEscape(apiKey))
	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("リクエスト作成失敗: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("APIリクエスト失敗: %w", err)
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return "", fmt.Errorf("API応答の読み込み失敗 (status %d): %w", resp.StatusCode, readErr)
	}
	if len(body) == 0 {
		return "", fmt.Errorf("Gemini API応答が空です (status %d)", resp.StatusCode)
	}

	var r GeminiGenerateContentResponse
	if err := json.Unmarshal(body, &r); err != nil {
		// JSON以外（HTMLなど）もあり得るので、状況が分かるように status と body を出す
		return "", fmt.Errorf("Gemini generateContent JSONパース失敗 (status %d, body=%q): %w", resp.StatusCode, string(body), err)
	}
	if resp.StatusCode != http.StatusOK {
		// bodyにerrorが含まれている場合はそちらを優先
		if r.Error != nil {
			return "", fmt.Errorf("Gemini APIエラー (status %d): %s", resp.StatusCode, r.Error.Message)
		}
		return "", fmt.Errorf("APIエラー (status %d): %s", resp.StatusCode, string(body))
	}
	if r.Error != nil {
		return "", fmt.Errorf("Gemini APIエラー: %s", r.Error.Message)
	}

	if len(r.Candidates) == 0 || len(r.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("応答が空です")
	}

	var sb strings.Builder
	for _, p := range r.Candidates[0].Content.Parts {
		if p.Text == "" {
			continue
		}
		sb.WriteString(p.Text)
	}

	text := strings.TrimSpace(sb.String())
	if text == "" {
		return "", fmt.Errorf("テキスト応答が見つかりません")
	}

	// 最終的に成功したバージョンを記録（デバッグ用）
	recordGeminiAPIVersionUsed(apiVersion)
	return text, nil
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

	newsModel := geminiModelFromEnv("GEMINI_MODEL_NEWS", "gemini-1.5-flash")
	enableSearch := geminiBoolFromEnv("GEMINI_ENABLE_GOOGLE_SEARCH", true)

	text, err := callGeminiGenerateContent(apiKey, newsModel, prompt, 1500, enableSearch)
	if err == nil {
		return text, nil
	}
	// Google Search toolが利用できない環境もあるため、フォールバックでツール無し再試行
	if enableSearch {
		fallbackText, fallbackErr := callGeminiGenerateContent(apiKey, newsModel, prompt, 1500, false)
		if fallbackErr == nil {
			return fallbackText, nil
		}
	}
	return "", err
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

		// VOICEVOXで読み上げ（ずんだもん: speakerID=3）
		if err := speakWithVoicevox(string(content), 3); err != nil {
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

	summaryModel := geminiModelFromEnv("GEMINI_MODEL_SUMMARY", "gemini-1.5-flash")
	// 要約は検索不要
	return callGeminiGenerateContent(apiKey, summaryModel, prompt, 300, false)
}

// DiffInfo 差分情報の構造体
type DiffInfo struct {
	FilePath    string
	FileName    string
	CompanyName string
	OldContent  string
	NewContent  string
}

// VOICEVOX API クライアント
const voicevoxBaseURL = "http://localhost:50021"

// speakWithVoicevox VOICEVOXで音声合成して読み上げる
func speakWithVoicevox(text string, speakerID int) error {
	// 1. 音声合成用のクエリを作成
	query, err := createAudioQuery(text, speakerID)
	if err != nil {
		return fmt.Errorf("音声クエリの作成に失敗: %w", err)
	}

	// 2. 音声を合成
	audioData, err := synthesis(query, speakerID)
	if err != nil {
		return fmt.Errorf("音声合成に失敗: %w", err)
	}

	// 3. 音声データを再生
	if err := playAudio(audioData); err != nil {
		return fmt.Errorf("音声再生に失敗: %w", err)
	}

	return nil
}

// createAudioQuery 音声合成用のクエリを作成
func createAudioQuery(text string, speakerID int) ([]byte, error) {
	apiURL := fmt.Sprintf("%s/audio_query?text=%s&speaker=%d",
		voicevoxBaseURL,
		url.QueryEscape(text),
		speakerID)

	resp, err := http.Post(apiURL, "application/json", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("APIエラー (status %d): %s", resp.StatusCode, string(body))
	}

	return io.ReadAll(resp.Body)
}

// synthesis 音声合成を実行
func synthesis(query []byte, speakerID int) ([]byte, error) {
	apiURL := fmt.Sprintf("%s/synthesis?speaker=%d", voicevoxBaseURL, speakerID)
	resp, err := http.Post(apiURL, "application/json", bytes.NewBuffer(query))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("APIエラー (status %d): %s", resp.StatusCode, string(body))
	}

	return io.ReadAll(resp.Body)
}

// playAudio 音声データを再生（macOSの場合）
func playAudio(audioData []byte) error {
	// 一時ファイルに保存
	tmpFile, err := os.CreateTemp("", "voicevox_*.wav")
	if err != nil {
		return err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(audioData); err != nil {
		return err
	}
	tmpFile.Close()

	// afplayコマンドで再生
	cmd := exec.Command("afplay", tmpFile.Name())
	return cmd.Run()
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

	// 差分がある場合は、読み上げ前にファイルに保存
	fmt.Printf("\n=== 差分情報をファイルに保存 ===\n")

	// diffsディレクトリを作成
	diffDir := "diffs"
	if err := os.MkdirAll(diffDir, 0755); err != nil {
		fmt.Printf("警告: diffsディレクトリの作成に失敗 - %v\n", err)
	}

	// タイムスタンプ付きのファイル名を生成
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	diffOutputPath := filepath.Join(diffDir, fmt.Sprintf("diff_%s.txt", timestamp))

	var diffOutput strings.Builder
	diffOutput.WriteString(fmt.Sprintf("差分検出日時: %s\n", time.Now().Format("2006-01-02 15:04:05")))
	diffOutput.WriteString(fmt.Sprintf("比較元: %s\n", oldDir))
	diffOutput.WriteString(fmt.Sprintf("比較先: %s\n", newDir))
	diffOutput.WriteString(fmt.Sprintf("差分ファイル数: %d\n\n", len(diffs)))
	diffOutput.WriteString("=== 差分詳細 ===\n\n")

	for i, diff := range diffs {
		diffOutput.WriteString(fmt.Sprintf("[%d] %s\n", i+1, diff.FileName))
		diffOutput.WriteString(fmt.Sprintf("会社名: %s\n", diff.CompanyName))
		if diff.OldContent == "" {
			diffOutput.WriteString("状態: 新規ファイル\n\n")
			diffOutput.WriteString("内容:\n")
			diffOutput.WriteString(diff.NewContent)
		} else {
			diffOutput.WriteString("状態: 更新\n\n")
			diffOutput.WriteString("【旧版】\n")
			diffOutput.WriteString(diff.OldContent)
			diffOutput.WriteString("\n\n【新版】\n")
			diffOutput.WriteString(diff.NewContent)
		}
		diffOutput.WriteString("\n\n" + strings.Repeat("=", 60) + "\n\n")
	}

	if err := os.WriteFile(diffOutputPath, []byte(diffOutput.String()), 0644); err != nil {
		fmt.Printf("警告: 差分ファイルの保存に失敗 - %v\n", err)
	} else {
		fmt.Printf("✓ 差分情報を保存しました: %s\n", diffOutputPath)
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

		// 要約を音声で読み上げ（ずんだもん: speakerID=3）
		fmt.Printf("  読み上げ中...\n")
		if err := speakWithVoicevox(summaryText, 3); err != nil {
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
	// Gemini APIキーを環境変数から取得
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("環境変数 GEMINI_API_KEY が設定されていません")
	}

	requestedVersion := geminiAPIVersionFromEnv()
	effectiveVersion := effectiveGeminiAPIVersion(requestedVersion)
	fmt.Printf("Gemini API version: requested=%s, effective=%s\n\n", requestedVersion, effectiveVersion)

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
	if v := getLastGeminiAPIVersionUsed(); v != "" {
		fmt.Printf("Gemini API version used (last success): %s\n", v)
	}

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
