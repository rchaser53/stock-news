package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
	"gopkg.in/yaml.v3"
)

// Company 会社情報の構造体
type Company struct {
	Name   string `yaml:"name"`
	Ticker string `yaml:"ticker"`
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

// getStockNews ChatGPT APIを使用して株価ニュースを取得する
func getStockNews(client *openai.Client, company Company) (string, error) {
	prompt := fmt.Sprintf(
		"%s（証券コード: %s）に関する最新の株価関連ニュースや重要な動向を3つ程度、簡潔に教えてください。",
		company.Name,
		company.Ticker,
	)

	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "あなたは株式市場の専門家です。最新の株価関連ニュースを簡潔に提供してください。",
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
			MaxTokens:   500,
			Temperature: 0.7,
		},
	)

	if err != nil {
		return "", fmt.Errorf("ChatGPT APIの呼び出しに失敗しました: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("ChatGPTからの応答がありません")
	}

	return resp.Choices[0].Message.Content, nil
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

	// OpenAIクライアントを作成
	client := openai.NewClient(apiKey)

	// 各会社のニュースを取得して出力
	fmt.Println("=== 株価関連ニュース ===")
	fmt.Println()

	for i, company := range config.Companies {
		fmt.Printf("[%d] %s (%s)\n", i+1, company.Name, company.Ticker)
		fmt.Println(strings.Repeat("-", 60))

		news, err := getStockNews(client, company)
		if err != nil {
			fmt.Printf("エラー: %v\n", err)
		} else {
			fmt.Println(news)
		}

		fmt.Println()
	}
}
