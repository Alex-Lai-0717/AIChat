from datetime import datetime
import openai
import chainlit as cl
import requests
import json
import asyncio
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
    user_session.set("selected_model", "gpt-3.5-turbo-16k-0613")
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
                values=["gpt-3.5-turbo-16k-0613", "vicuna"],
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
    weather_triggers = ["今天的天氣", "明天的天氣", "桃園天氣如何", "桃園會下雨嗎","今天天氣","明天天氣"]
    if any(trigger in message for trigger in weather_triggers):
        weather_report = get_weather_report()
        await cl.Message(content=weather_report).send()
        return

    elif message == "上傳檔案":
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
    timeout_count = 0
    while timeout_count < 2:
        files = cl.AskFileMessage(
            content="開始上傳檔案!", accept={"text/plain": [".txt", ".py", ".csv"]}, max_size_mb=100, max_files=10,
            timeout=5, raise_on_timeout=True
        )
        done, pending = await asyncio.wait([files.send(), countdown_timer(files, 5)],
                                           return_when=asyncio.FIRST_COMPLETED)
        response_task = next(iter(done))
        response = await response_task
        if response == 'TIMEOUT':
            await handle_timeout(timeout_count)
            timeout_count += 1
        else:
            for task in pending:
                task.cancel()
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


async def countdown_timer(res, seconds):
    for remaining_time in range(seconds, 0, -1):
        res.content = f"開始上傳檔案! {remaining_time}秒"
        await res.update()
        await asyncio.sleep(1)
    return 'TIMEOUT'


async def handle_timeout(count):
    messages = [
        f"檔案上傳已超時，請重新上傳檔案",
        f"警告：上傳已超時兩次，上傳指令將結束"
    ]
    if count < len(messages):
        await cl.Message(content=messages[count]).send()


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

def get_weather_data(url, config):
    """從 API 獲取天氣數據。"""
    try:
        response = requests.get(url, params=config)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to get data from API: {e}")
        return None

def process_weather_data(data):
    """處理原始天氣數據，轉換成更易使用的格式。"""
    location_data = data["records"]["location"][0]
    elements = location_data["weatherElement"]

    times = set()
    data_by_time = {}

    for element in elements:
        for time in element["time"]:
            start_time = time["startTime"]
            end_time = time["endTime"]
            times.add((start_time, end_time))
            data_by_time[(start_time, end_time)] = {}

    for element in elements:
        element_name = element["elementName"]
        element_name = translate_element_name(element_name)

        for time in element["time"]:
            start_time = time["startTime"]
            end_time = time["endTime"]
            parameter = time["parameter"]
            parameter_name = process_parameter(parameter, element_name)

            data_by_time[(start_time, end_time)][element_name] = parameter_name

    return location_data, times, data_by_time

def translate_element_name(element_name):
    """將元素名稱翻譯成中文。"""
    translations = {
        "Wx": "氣候",
        "PoP": "降雨機率",
        "MinT": "最低溫度",
        "MaxT": "最高溫度",
        "CI": "體感程度",
    }
    return translations.get(element_name, element_name)

def process_parameter(parameter, element_name):
    """處理參數，使其格式更易讀。"""
    parameter_name = parameter["parameterName"]
    parameter_unit = parameter.get("parameterUnit")

    if parameter_unit:
        parameter_name += parameter_unit

    if element_name == "降雨機率" and "百分比" in parameter_name:
        parameter_name = parameter_name.replace("百分比", "%")

    if element_name in ["最高溫度", "最低溫度"] and "C" in parameter_name:
        parameter_name = parameter_name.replace("C", "℃")

    return parameter_name

def load_config(config_path):
    """從一個 json 文件中讀取配置。"""
    with open(config_path, 'r', encoding="utf-8") as f:
        config = json.load(f)
    return config

def format_time_str(start_time, end_time):
    """格式化時間字串。"""
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    if start_time.date() == end_time.date():
        time_str = f"{start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%H:%M')}"
    else:
        time_str = f"{start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')}"
    return time_str

def get_weather_report():
    """從 API 獲取天氣報告並返回。"""
    config = load_config("config.json")
    data = get_weather_data(config['URL'], config)
    if data is None:
        return "對不起，無法從 API 獲取天氣數據。"

    location_data, times, data_by_time = process_weather_data(data)
    report = []

    element_order = ["氣候", "降雨機率", "體感程度"]
    report.append(f'{location_data["locationName"]} {data["records"]["datasetDescription"]}\n')

    for time in sorted(times):
        time_data = []
        start_time_str, end_time_str = time
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        time_str = format_time_str(start_time, end_time)
        time_data.append(f"時間: {time_str}")

        for element_name in element_order:
            parameter_name = data_by_time[time].get(element_name)
            if parameter_name is not None:
                time_data.append(f"{element_name}: {parameter_name}")

        min_temp = data_by_time[time].get("最低溫度")
        max_temp = data_by_time[time].get("最高溫度")
        if min_temp is not None and max_temp is not None:
            time_data.append(f"溫度: {min_temp} ~ {max_temp}")

        report.append("\n".join(time_data))

    return "\n".join(report)


async def get_ai_response(model_name, message_history):
    """根據所選模型獲取 AI 的響應。"""
    if model_name == "vicuna":
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "vicuna",
            "messages": message_history,
            "max_tokens": 1500,
        }
        try:
            response = requests.post(URL_TO_VICUNA, headers=headers, data=json.dumps(data), timeout=60)
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            # Handle error as appropriate
            print(f"Request error: {e}")
            return "獲取 AI 回覆時發生錯誤。"
    else:
        msg = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
                model=model_name, messages=message_history, stream=True, **OPENAI_SETTINGS
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
