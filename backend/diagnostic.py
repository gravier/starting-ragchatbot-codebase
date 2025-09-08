#!/usr/bin/env python3
"""
Diagnostic script to identify RAG system issues without pytest dependency
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test configuration for issues"""
    print("=== CONFIGURATION DIAGNOSTICS ===")
    
    try:
        from config import Config
        config = Config()
        
        issues = []
        
        print(f"ANTHROPIC_API_KEY: {'[SET]' if config.ANTHROPIC_API_KEY else '[MISSING]'}")
        print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
        print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
        print(f"CHROMA_PATH: {config.CHROMA_PATH}")
        
        # Check for invalid model name
        if config.ANTHROPIC_MODEL.startswith("1"):
            issues.append(f"❌ Invalid model name: {config.ANTHROPIC_MODEL} (has '1' prefix)")
        else:
            issues.append(f"✅ Model name looks valid: {config.ANTHROPIC_MODEL}")
        
        # Check for missing API key
        if not config.ANTHROPIC_API_KEY:
            issues.append("❌ Missing ANTHROPIC_API_KEY")
        else:
            issues.append("✅ API key is set")
        
        # Check for paths
        chroma_dir = os.path.dirname(config.CHROMA_PATH) if config.CHROMA_PATH else ""
        if chroma_dir and not os.path.exists(chroma_dir):
            issues.append(f"❌ ChromaDB directory does not exist: {chroma_dir}")
        else:
            issues.append(f"✅ ChromaDB path exists: {config.CHROMA_PATH}")
        
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  {issue}")
        
        return len([i for i in issues if i.startswith("❌")]) == 0
    
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("\n=== IMPORT DIAGNOSTICS ===")
    
    modules = [
        'vector_store',
        'search_tools', 
        'ai_generator',
        'rag_system',
        'models',
        'config'
    ]
    
    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            success = False
    
    return success

def test_course_search_tool():
    """Test CourseSearchTool functionality"""
    print("\n=== COURSE SEARCH TOOL DIAGNOSTICS ===")
    
    try:
        from search_tools import CourseSearchTool
        from unittest.mock import Mock
        from vector_store import SearchResults
        
        # Create mock vector store
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Test content about MCP"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        # Test tool
        tool = CourseSearchTool(mock_store)
        
        # Test tool definition
        definition = tool.get_tool_definition()
        print(f"✅ Tool definition created: {definition['name']}")
        
        # Test execution
        result = tool.execute("test query")
        print(f"✅ Tool execution successful: {len(result)} chars returned")
        
        return True
    
    except Exception as e:
        print(f"❌ CourseSearchTool test failed: {e}")
        traceback.print_exc()
        return False

def test_ai_generator():
    """Test AIGenerator basic functionality"""
    print("\n=== AI GENERATOR DIAGNOSTICS ===")
    
    try:
        from ai_generator import AIGenerator
        
        # Test initialization (without actual API calls)
        generator = AIGenerator("test-key", "test-model")
        print(f"✅ AIGenerator initialization successful")
        
        # Test system prompt
        if "search_course_content" in generator.SYSTEM_PROMPT:
            print("✅ System prompt contains tool references")
        else:
            print("❌ System prompt missing tool references")
        
        return True
    
    except Exception as e:
        print(f"❌ AIGenerator test failed: {e}")
        traceback.print_exc()
        return False

def test_rag_system_init():
    """Test RAGSystem initialization"""
    print("\n=== RAG SYSTEM INITIALIZATION DIAGNOSTICS ===")
    
    try:
        from rag_system import RAGSystem
        from config import Config
        from unittest.mock import patch, Mock
        
        config = Config()
        
        # Mock external dependencies to avoid API calls
        with patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai:
            
            mock_vs.return_value = Mock()
            mock_ai.return_value = Mock()
            
            rag_system = RAGSystem(config)
            print("✅ RAGSystem initialization successful")
            
            # Test tool registration
            tools = rag_system.tool_manager.get_tool_definitions()
            print(f"✅ Tools registered: {len(tools)} tools")
            
            if len(tools) == 2:
                tool_names = [t['name'] for t in tools]
                print(f"   Tools: {', '.join(tool_names)}")
            else:
                print(f"❌ Expected 2 tools, got {len(tools)}")
                return False
            
            return True
    
    except Exception as e:
        print(f"❌ RAGSystem initialization failed: {e}")
        traceback.print_exc()
        return False

def test_vector_store_data():
    """Test if vector store has data"""
    print("\n=== VECTOR STORE DATA DIAGNOSTICS ===")
    
    try:
        from vector_store import VectorStore
        from config import Config
        
        config = Config()
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check if there's any course data
        course_count = vector_store.get_course_count()
        course_titles = vector_store.get_existing_course_titles()
        
        print(f"Course count: {course_count}")
        print(f"Course titles: {course_titles}")
        
        if course_count == 0:
            print("❌ No courses loaded in vector store - this will cause search failures")
            return False
        else:
            print(f"✅ {course_count} courses loaded")
            return True
    
    except Exception as e:
        print(f"❌ Vector store data test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostics"""
    print("Starting RAG System Diagnostics...\n")
    
    results = {
        "Configuration": test_configuration(),
        "Imports": test_imports(),
        "CourseSearchTool": test_course_search_tool(),
        "AIGenerator": test_ai_generator(),
        "RAGSystem Init": test_rag_system_init(),
        "Vector Store Data": test_vector_store_data()
    }
    
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    failed_tests = []
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            failed_tests.append(test_name)
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        print("\nLikely causes of 'query failed' error:")
        
        if "Configuration" in failed_tests:
            print("- Invalid model name or missing API key")
        if "Vector Store Data" in failed_tests:
            print("- No course data loaded - searches will return empty results")
        if "AIGenerator" in failed_tests:
            print("- AI generator setup issues")
        if "RAGSystem Init" in failed_tests:
            print("- System initialization problems")
    else:
        print("\n✅ All diagnostics passed!")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)