import openai
import chainlit as cl
import requests
import json
from chainlit.input_widget import Select, Switch
from chainlit import user_session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

URL_TO_VICUNA = "http://10.30.9.17:8000/v1/chat/completions"
OPENAI_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
SYSTEM_TEMPLATE = """
首先所有回答都"必須使用繁體中文"
使用以下的文本的上下文來回答用戶的問題。
如果你不知道答案，直接說你不知道，不要嘗試編造答案。
所有回答都"必須"使用繁體中文
你的答案中必須始終包含一個"來源"部分。
"來源"部分應該是你得到答案的文檔的參考來源。
你的回答應該像這樣：


```
答案是 foo
來源: xyz
```

開始!
----------------
{summaries}
"""
messages = [SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}"), ]
prompt = ChatPromptTemplate.from_messages(messages)
CHAIN_TYPE_KWARGS = {"prompt": prompt}


@cl.on_chat_start
async def init():
    """初始化聊天室."""
    user_session.set("continuous_mode", True)
    user_session.set(
        "message_history",
        [{"role": "system", "content": "連續對話，啟動！！！"}],
    )
    await cl.Message(content="【公告】歡迎使用本聊天室，"
                             "聊天室相關設定可以於輸入框左側的設定調整").send()
    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="GPT - Model",
                values=["gpt-3.5-turbo", "vicuna"],
                initial_index=0,
            ),
            Switch(id="continuous", label="預設為連續對話，關閉按鈕切換為單次對話", initial=True),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """根據用戶設置設置 AI 代理。."""
    selected_model = settings["Model"]
    continuous_mode = settings["continuous"]
    await cl.Message(content=f"當前model為{selected_model},連續對話mode為{continuous_mode}").send()
    user_session.set("selected_model", selected_model)
    user_session.set("continuous_mode", continuous_mode)


@cl.on_message
async def handle_message(message: str):
    """處理用戶的訊息."""
    continuous_mode = user_session.get("continuous_mode")
    if message == "上傳檔案":
        await upload_file()
        return
    elif "我想問" in message:
        cleaned_message = message.replace("我想問", "用中文回答我", 1)
        await handle_file_query(cleaned_message.strip())
        return
    elif continuous_mode:
        await continuous_chat(message)
    else:
        await single_chat(message)


async def upload_file():
    """允許用戶上傳文件並處理"""
    files = await cl.AskFileMessage(
        content="開始上傳檔案!", accept={"text/plain": [".txt", ".py"]}, max_size_mb=100, max_files=10
    ).send()
    file = files[0]
    msg = cl.Message(content=f"正在上傳 `{file.name}`......")
    await msg.send()
    text = file.content.decode("utf-8")
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    user_session.set("metadatas", metadatas)
    user_session.set("texts", texts)
    msg.content = f"`{file.name}` 已上傳成功. 請告訴我有哪些地方可以協助你!"
    await msg.update()
    user_session.set("chain", chain)


async def handle_file_query(message):
    """處理與上傳文件相關的查詢。."""
    chain = user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []
    metadatas = user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = user_session.get("texts")
    if sources:
        found_sources = [
            source.strip().replace(".", "")
            for source in sources.split(",")
            if source.strip().replace(".", "") in all_sources
        ]
        for source_name in found_sources:
            index = all_sources.index(source_name)
            text = texts[index]
            source_elements.append(cl.Text(content=text, name=source_name))
        answer += f"\nSources: {', '.join(found_sources)}"
    else:
        answer += "\nNo sources found"
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()


async def get_ai_response(model_name, message_history):
    """根據所選模型獲取 AI 的響應。"""
    if model_name == "vicuna":
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "vicuna-33b-v1.3",
            "messages": message_history,
            "max_tokens": 1500,
        }
        response = requests.post(URL_TO_VICUNA, headers=headers, data=json.dumps(data), timeout=60)
        return response.json()["choices"][0]["message"]["content"]
    else:
        msg = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo", messages=message_history, stream=True, **OPENAI_SETTINGS
        ):
            token = stream_resp.choices[0]["delta"].get("content", "")
            await msg.stream_token(token)
        return msg


async def continuous_chat(message):
    """處理連續聊天模式。"""
    message_history = user_session.get("message_history")
    selected_model = user_session.get("selected_model")
    message_history.append({"role": "user", "content": message})
    response = await get_ai_response(selected_model, message_history)
    if isinstance(response, cl.Message):
        await response.send()
        message_history.append({"role": "assistant", "content": response.content})
    else:
        await cl.Message(content=response).send()
        message_history.append({"role": "assistant", "content": response})


async def single_chat(message):
    """處理單次聊天模式。"""
    selected_model = user_session.get("selected_model")
    mes = [{"role": "user", "content": message}]
    response = await get_ai_response(selected_model, mes)
    if isinstance(response, cl.Message):
        await response.send()
    else:
        await cl.Message(content=response).send()
