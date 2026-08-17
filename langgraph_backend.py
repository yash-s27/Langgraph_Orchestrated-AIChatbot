from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,Any, Dict, Optional,Iterator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from tavily import TavilyClient
import requests
import os
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
from pypdf import PdfReader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
embbeding_engine = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True} 
)


#--------pdf retrive store per thread
_THREAD_RETRIEVERS : Dict[str, Any] = {}
_THREAD_METADATA  : Dict[str, dict] = {}

class PDFloader(BaseLoader):

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """Loads pages one by one safely extracting strings."""
        reader = PdfReader(self.file_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            yield Document(
                page_content=text,
                metadata={"source": self.file_path, "page": page_num + 1},
            )

def _get_retriever(thread_id : Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None



def ingest_pdf(file_bytes : bytes, thread_id : str, filename : Optional[str] = None)-> dict:
    """
    Build a FAISS reteriever for the uploaded PDF and store it for thr thread.
    Return a summary dict thst can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes recived for ingestion")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PDFloader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            separators=['\n\n', '\n', ' ', ''] 
        )

        chunks = splitter.split_documents(docs)

        vector_store =  Chroma.from_documents(
            documents= chunks,
            embedding=embbeding_engine,
            persist_directory= f"./chromaDB/thread_{thread_id}"
        )
        retriever = vector_store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {'k':4}
        )
    
        display_name = filename or os.path.basename(temp_path)
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": display_name,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": display_name,
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        # Step 6: Hard drive cleanup is mandatory to avoid storage leaks
        try:
            os.remove(temp_path)
        except OSError:
            pass
@tool
def search_tool(query: str) -> str:
    """
    Search the live internet for recent news, events, and facts.
    Use this tool whenever the user asks about real-time or current information.
    and be specific about the date and time when the tool executed fetch live data not the data of cut off date of llm
    if this tool return none then dont show error insted rely on llm answer.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: Missing TAVILY_API_KEY in environment variables."
        
    try:
    
        tavily_client = TavilyClient(api_key=api_key)
        
        search_context = tavily_client.search(
            query=query,
            search_depth='advanced',
            max_results=3
        )
        
        return str(search_context)
        
    except Exception as e:
        return f"An error occurred while executing the search: {str(e)}"


@tool
def calculator(first_num: float, second_num: float, operation :str)->dict:
    """ 
    Perform a basic athematic operation on two numbers.
    Supported Operation : add, sub, mul, div,
    For Methimatical opertion of two number use this tool
    """
    try:
        if operation == 'add':
            res = first_num + second_num
        elif operation =='sub':
            res = first_num - second_num
        elif operation == 'mul':
            res = first_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            res = first_num / second_num
        else:
            return {'error':f'Unsupported operation {operation}'}
        return {"first number":first_num,
                "second_num": second_num,
                "result":res}
    except Exception as e:
        return {'error': e}


@tool
def get_stock_price(symbol : str)->dict:
    """
    Fetch latest stock price for a givin symbol (example : 'AAPL', 'TSLA')
    using ALpha vantage using API key in the URl
    Convert the currency in INR along with showing actuall currency.
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv("ALPHA_VINTAGE_API_KEY")}'
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query:str, congig : RunnableConfig) -> dict:
    """
    Search and retrieve deep analytical context from the uploaded PDF document 
    specific to this active thread session.
    do not share information with other thread/chat session
    """
    configurabel = congig.get('configurable', {})
    thread_id = configurabel.get('thread_id')

    if not thread_id:
        return {"error": "System Error: Missing active conversational thread reference context."}

    #checking memory resgister
    retriever = _THREAD_RETRIEVERS.get(str(thread_id))

        # 3. IF Server Restarted: Re-hydrate instantly from disk safely! (Self-Healing)
    if retriever is None:
        db_path = f"./chroma_db/thread_{thread_id}"
        if os.path.exists(db_path):
            # Reinitialize the database pointer using your production HuggingFaceEmbeddings instance
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function= embbeding_engine  # Globally initialized HuggingFace model
            )
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            # Cache it back to memory register for faster subsequent turns
            _THREAD_RETRIEVERS[str(thread_id)] = retriever
        else:
            return {
                "error": "No indexed knowledge-base repository detected for this workspace. Please upload a PDF first.",
                "query": query
            }
            
    # 4. Standard context execution metrics parsing
    try:
        retrieved_documents = retriever.invoke(query)
        
        context_blocks = [doc.page_content for doc in retrieved_documents]
        metadata_blocks = [doc.metadata for doc in retrieved_documents]
        
        # Pull original tracking attributes safely
        session_meta = _THREAD_METADATA.get(str(thread_id), {})
        
        return {
            "query": query,
            "context": context_blocks,
            "metadata": metadata_blocks,
            "source_file": session_meta.get("filename", "Extracted_Local_Database")
        }
    except Exception as e:
        return {"error": f"Retrieval interface failure exception layer: {str(e)}"}

@tool
def github_user_summary(username: str) -> dict:
    """
    Fetch a GitHub user's public profile and repositories, then generate a
    recruiter-style skills summary — useful for freshers/students showcasing
    a project portfolio, where repos may have 0 stars but still demonstrate
    real skills. Use when the user asks things like 'summarize this person's
    GitHub' or 'what skills does <username> have'.
    No authentication required (GitHub's public API, ~60 requests/hour).
    if there are question like tell me about then a github username then also use this tool.
    """
    try:
        profile_r = requests.get(f"https://api.github.com/users/{username}", timeout=10)
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    if profile_r.status_code != 200:
        return {"error": f"User not found or API error ({profile_r.status_code})"}
    profile = profile_r.json()

    try:
        repos_r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"sort": "updated", "per_page": 30},
            timeout=10,
        )
        repos = repos_r.json() if repos_r.status_code == 200 else []
    except Exception:
        repos = []

    # Skip forks, keep original repos only — do NOT filter/sort by stars,
    # since freshers' projects are often 0-star but still show real skill.
    # "updated" sort from the API already puts most recently worked-on repos first.
    own_repos = [r for r in repos if isinstance(r, dict) and not r.get("fork")]
    top_repos = own_repos[:15]

    languages: Dict[str, int] = {}
    for r in top_repos:
        lang = r.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1

    repo_lines = [
        f"- {r.get('name')}: {r.get('description') or 'no description'} "
        f"(language: {r.get('language') or 'unknown'}, last updated: {r.get('updated_at')})"
        for r in top_repos
    ]

    prompt = f"""
You are writing a recruiter-facing summary of a developer's GitHub profile.
This person may be a student or fresher — judge them on what their projects
DO and what skills they demonstrate, not on stars or popularity. A 0-star
repo that builds a full-stack app or an ML pipeline still shows real skill.

Profile:
- Name: {profile.get('name') or username}
- Bio: {profile.get('bio') or 'N/A'}
- Public repos: {profile.get('public_repos')}
- Company: {profile.get('company') or 'N/A'}
- Location: {profile.get('location') or 'N/A'}

Repositories (most recently active first, regardless of star count):
{chr(10).join(repo_lines) if repo_lines else 'No public repositories found.'}

Languages used across these repos: {', '.join(languages.keys()) if languages else 'Unknown'}

Write a recruiter-style summary with:
1. A 2-3 sentence overview of the person's technical profile and what stage
   they seem to be at (e.g. learning fundamentals, building full projects,
   specializing in a domain).
2. A "Core skills" list (languages, frameworks, domains — inferred from repo
   names/descriptions, not from popularity).
3. A "Notable projects" section: 3-5 repos, each with ONE line on what it
   demonstrates (e.g. "expense-tracker — built a CRUD app with auth using
   Flask and PostgreSQL"). Include smaller/0-star repos if they show a
   skill not shown elsewhere.

Be factual and don't overstate based on limited data. Keep it concise and scannable.
""".strip()

    try:
        summary_response = llm.invoke(prompt)
        summary_text = summary_response.content
    except Exception as e:
        summary_text = f"Could not generate summary: {str(e)}"

    return {
        "username": username,
        "name": profile.get("name"),
        "public_repos": profile.get("public_repos"),
        "top_languages": list(languages.keys()),
        "recruiter_summary": summary_text,
    }

import math

@tool
def trip_itinerary_planner(origin: str, destination: str, days: int = 3, interests: str = "") -> dict:
    """
    Plan a full day-by-day trip itinerary from an origin to a destination,
    like a professional trip planner would. Takes into account real distance
    and current weather at the destination. Use when the user asks to plan
    a trip, vacation, or itinerary between two places.
    'interests' is optional (e.g. 'food, history, nightlife').
    use this tool whenever user ask plan a trip,  create an itinerary, plan for a holiday et type of queries use this tool as priority rather than general llm response.
    dont get confuse in place names if you need search tool for having accurate name which can be used in this tool so use search tool but be spicific about origin and destination place name.
    """
    headers = {"User-Agent": "portfolio-trip-planner-app"}

    def geocode(place: str):
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": place, "format": "json", "limit": 1},
                headers=headers,
                timeout=10,
            )
            data = r.json()
            if not data:
                return None
            return {
                "display_name": data[0]["display_name"],
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
            }
        except Exception:
            return None

    origin_geo = geocode(origin)
    dest_geo = geocode(destination)

    if not origin_geo:
        return {"error": f"Could not find location: {origin}"}
    if not dest_geo:
        return {"error": f"Could not find location: {destination}"}

    # Haversine distance (km) — no API needed
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    distance_km = round(
        haversine(origin_geo["lat"], origin_geo["lon"], dest_geo["lat"], dest_geo["lon"]), 1
    )

    # Weather forecast at destination (Open-Meteo, free, no key)
    weather_summary = "Weather data unavailable."
    try:
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": dest_geo["lat"],
                "longitude": dest_geo["lon"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": min(days, 16),
                "timezone": "auto",
            },
            timeout=10,
        )
        wd = w.json().get("daily", {})
        if wd:
            lines = []
            for i, date in enumerate(wd.get("time", [])):
                lines.append(
                    f"{date}: {wd['temperature_2m_min'][i]}–{wd['temperature_2m_max'][i]}°C, "
                    f"{wd['precipitation_probability_max'][i]}% chance of rain"
                )
            weather_summary = "\n".join(lines)
    except Exception:
        pass

    prompt = f"""
You are a professional trip planner. Create a detailed, practical day-by-day
itinerary for a trip.

Origin: {origin_geo['display_name']}
Destination: {dest_geo['display_name']}
Straight-line distance: {distance_km} km
Trip length: {days} day(s)
Traveler interests: {interests or 'general sightseeing, not specified'}

Forecasted weather at destination:
{weather_summary}

Write the itinerary as:
1. A short intro (2-3 sentences) — best way to travel given the distance
   (flight/train/road), and general trip vibe.
2. Day-by-day plan (Day 1, Day 2, ...): morning / afternoon / evening
   activities, suggested local food to try, and one practical tip per day.
3. Adjust suggestions sensibly based on the weather (e.g. indoor activities
   on rainy days).
4. A short "Packing tips" section based on the weather.

Be concrete and use real place-type suggestions (e.g. "visit the old town
market", "a riverside walk") even if you don't know exact venue names. Keep
it well-organized and scannable.
""".strip()

    try:
        response = llm.invoke(prompt)
        itinerary_text = response.content
    except Exception as e:
        itinerary_text = f"Could not generate itinerary: {str(e)}"

    return {
        "origin": origin_geo["display_name"],
        "destination": dest_geo["display_name"],
        "distance_km": distance_km,
        "days": days,
        "weather_forecast": weather_summary,
        "itinerary": itinerary_text,
    }

tools = [search_tool, get_stock_price, calculator, rag_tool, github_user_summary, trip_itinerary_planner]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chatnode(state : ChatState) -> ChatState:
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages' : [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread = False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chatnode', chatnode)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chatnode')
graph.add_conditional_edges('chatnode', tools_condition)
graph.add_edge('tools', 'chatnode')
graph.add_edge('chatnode', END)

chatbot = graph.compile(checkpointer=checkpointer)

# geeting all threads from sqlite database by checkpointer.list()
def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


def delete_thread(thread_id: str) -> None:
    thread_id = str(thread_id)
    cur = conn.cursor()
    cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()
    cur.close()
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)


def thread_document_metadata(thread_id: str) -> Optional[dict]:
    """Surfaces the uploaded PDF stats (filename, chunks, pages) for a specific thread to the UI."""
    # Global tracking register se data nikal kar safe form mein bhejta hai
    return _THREAD_METADATA.get(str(thread_id), None)


def get_thread_title(thread_id: str, max_len: int = 40) -> str:
    """
    Derives a human-friendly chat title from the first user message stored
    in the checkpointer for this thread (similar to how ChatGPT titles chats).
    Falls back to 'New Chat' if no user message exists yet.
    """
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
        messages = state.values.get("messages", [])
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = (msg.content or "").strip().replace("\n", " ")
                if not text:
                    continue
                if len(text) > max_len:
                    text = text[:max_len].rstrip() + "…"
                return text
    except Exception:
        pass
    return "New Chat"