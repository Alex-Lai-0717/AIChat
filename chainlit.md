# Chainlit AIChat 🚀🤖
- **這是一個基於 Chainlit 和 OpenAI 的聊天機器人應用。它允許用戶與機器人進行連續或單次對話，並支持上傳文檔來獲得具體答案。**

## 特點
- **連續和單次對話模式:** 用戶可以選擇與機器人進行連續或單次的對話。
- **文檔上傳:** 用戶可以上傳文檔，並根據文檔內容向機器人提問。
- **多模型支持:** 目前支持 OpenAI 的 gpt-3.5-turbo 和 Vicuna 模型。

## 使用方法
- **啟動 Chainlit 應用。**
- **選擇你想要的模型和對話模式。**
- **輸入你的問題或上傳文檔進行詢問。**

## requirements.txt
- openai
- chainlit
- requests
- langchain

## 配置
- Vicuna 伺服器地址: url_to_vicuna
- 對話設置: settings

## 啟動
使用以下命令啟動應用：
python your_script_name.py