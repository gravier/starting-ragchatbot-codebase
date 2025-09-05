# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `./run.sh` (starts FastAPI server on port 8000)
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Dependencies**: `uv sync` (install Python dependencies)

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_api_key_here`
- Application serves frontend at `http://localhost:8000`
- API docs available at `http://localhost:8000/docs`

## RAG System Architecture

This is a Retrieval-Augmented Generation (RAG) chatbot system with a tool-based architecture where Claude API uses function calling to search course content when needed.

### Core Flow
1. **Frontend** (`frontend/script.js`) → **FastAPI** (`backend/app.py`) → **RAGSystem** (`backend/rag_system.py`)
2. **RAGSystem** calls **AIGenerator** (`backend/ai_generator.py`) with tool definitions
3. **Claude API** decides whether to use search tools or answer directly
4. If search needed: **CourseSearchTool** → **VectorStore** → **ChromaDB**
5. **Claude API** generates contextual response using search results

### Key Components

**RAGSystem (`rag_system.py`)**: Main orchestrator that coordinates all components. Handles document ingestion, query processing, and response generation.

**AIGenerator (`ai_generator.py`)**: Manages Claude API interactions with tool support. Uses a specialized system prompt optimized for educational content with strict tool usage guidelines.

**Tool System (`search_tools.py`)**: Implements Anthropic's function calling pattern. `CourseSearchTool` provides semantic search with course name matching and lesson filtering.

**VectorStore (`vector_store.py`)**: ChromaDB wrapper with separate collections for course metadata and content chunks. Uses SentenceTransformer embeddings ("all-MiniLM-L6-v2").

**SessionManager (`session_manager.py`)**: Maintains conversation history with configurable message limits for context continuity.

**DocumentProcessor (`document_processor.py`)**: Processes course documents into structured `Course` and `CourseChunk` objects with intelligent text chunking.

### Data Models (`models.py`)
- `Course`: Course metadata with lessons list
- `Lesson`: Individual lessons with numbers and titles  
- `CourseChunk`: Text chunks for vector storage with course/lesson references

### Configuration (`config.py`)
- Centralized settings using dataclass pattern
- Environment variable loading with sensible defaults
- Key settings: chunk size (800), overlap (100), max results (5), conversation history (2)

## Important Development Notes

### Document Processing
- Course documents in `docs/` folder are auto-loaded on startup
- Documents are chunked using sentence-based splitting with overlap
- Course titles serve as unique identifiers - avoid duplicates
- Supported formats: PDF, DOCX, TXT

### Vector Storage
- ChromaDB persistence at `./chroma_db` 
- Separate collections for course metadata vs content chunks
- Automatic deduplication based on course titles
- Search results include distances and metadata for relevance scoring

### Tool System Architecture
- Claude API uses function calling to determine when to search
- Single search per query maximum to control API costs
- Tool results are processed and synthesized by Claude, not returned raw
- Sources tracking happens at tool level, not vector store level

### Session Management
- Sessions auto-created if not provided in API requests
- Conversation history is trimmed to stay within token limits
- History provides context for follow-up questions

### Frontend Integration
- Uses relative API paths (`/api`) for deployment flexibility
- Loading states and error handling for better UX
- Markdown rendering for assistant responses
- Collapsible source citations when available

## File References
- Main entry: `backend/app.py:66` (query processing)
- RAG orchestration: `backend/rag_system.py:102` (main query method)
- AI generation: `backend/ai_generator.py:43` (response generation with tools)
- Course search: `backend/search_tools.py:20` (semantic search implementation)
- Vector operations: `backend/vector_store.py:34` (ChromaDB interface)