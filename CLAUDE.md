# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# One-time setup
uv sync

# Start the server (from project root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

Always use `uv run` to execute Python — never invoke `python` or `pip` directly.

There is no test suite.

## Architecture

This is a full-stack RAG chatbot. The backend is a single FastAPI process; the frontend is static HTML/JS served by that same process.

### Query Flow

1. `frontend/script.js` POSTs `{ query, session_id }` to `/api/query`
2. `backend/app.py` creates a session if needed, calls `RAGSystem.query()`
3. `backend/rag_system.py` fetches conversation history, then calls `AIGenerator.generate_response()` with the `search_course_content` tool available
4. `backend/ai_generator.py` makes a first Claude API call. If Claude decides to search (stop_reason `tool_use`), it calls `CourseSearchTool.execute()`, feeds the results back, and makes a second Claude API call to synthesize the final answer
5. `backend/search_tools.py` executes the search via `VectorStore.search()` and tracks sources for the UI
6. Sources and response are returned through `rag_system.py` → `app.py` → frontend

### Document Ingestion Flow

At startup, `app.py` calls `rag_system.add_course_folder("../docs")`:
1. `DocumentProcessor.process_course_document()` parses each `.txt/.pdf/.docx` file — expects a header of `Course Title`, `Course Link`, `Course Instructor` followed by `Lesson N: Title` markers
2. Lesson content is split into sentence-based chunks (800 chars, 100-char overlap)
3. `VectorStore` stores course metadata in the `course_catalog` ChromaDB collection and text chunks in `course_content`
4. Already-indexed courses are skipped (checked by title)

### Key Configuration (`backend/config.py`)

| Setting | Default | Effect |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model used for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `MAX_RESULTS` | `5` | Top-K chunks returned per search |
| `MAX_HISTORY` | `2` | Conversation exchanges kept per session |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory (relative to `backend/`) |

### Adding a New Tool

1. Subclass `Tool` in `backend/search_tools.py`, implement `get_tool_definition()` and `execute()`
2. Register it in `RAGSystem.__init__()` via `self.tool_manager.register_tool(your_tool)`
3. The tool will automatically be included in Claude's tool list and callable during generation

### Course Document Format

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <title>
...
```
