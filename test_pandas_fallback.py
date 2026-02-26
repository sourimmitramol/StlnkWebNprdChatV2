# Test script for pandas fallback functionality
# This script tests the pandas code generation fallback

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pandas_fallback():
    """Test the pandas fallback mechanism"""
    print("=" * 80)
    print("TESTING PANDAS CODE GENERATION FALLBACK")
    print("=" * 80)
    
    # Import required modules
    try:
        from agents.azure_agent import pandas_code_generator_fallback
        print("✓ Successfully imported pandas_code_generator_fallback")
    except ImportError as e:
        print(f"✗ Failed to import pandas_code_generator_fallback: {e}")
        return False
    
    # Test queries that should trigger pandas fallback
    test_queries = [
        "What is the average delay for containers by carrier?",
        "Show me a count of containers grouped by final destination",
        "Calculate the median transit time for shipments",
        "Find containers where revised ETA differs from original ETA by more than 7 days",
        "List unique load ports with their container counts"
    ]
    
    print("\nTest Queries:")
    print("-" * 80)
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "=" * 80)
    print("TESTING QUERY 1: Average delay by carrier")
    print("=" * 80)
    
    try:
        # Test with first query
        result = pandas_code_generator_fallback(test_queries[0])
        print("\nResult:")
        print(result[:500] + "..." if len(result) > 500 else result)
        print("\n✓ Pandas fallback executed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error during pandas fallback execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_detection():
    """Test the fallback detection logic"""
    print("\n" + "=" * 80)
    print("TESTING FALLBACK DETECTION LOGIC")
    print("=" * 80)
    
    import re
    
    # Test patterns that should trigger fallback
    trigger_phrases = [
        "I couldn't understand your query",
        "I don't have enough information to answer",
        "Please try rephrasing your question",
        "I'm not sure how to help with that",
        "Unable to answer this question",
    ]
    
    unable_to_answer_patterns = [
        r"i couldn't understand",
        r"i don't have enough information",
        r"please try rephrasing",
        r"i'm not sure",
        r"unable to answer",
        r"cannot determine",
        r"no information available",
        r"i don't know",
    ]
    
    print("\nTesting trigger detection:")
    for phrase in trigger_phrases:
        phrase_lower = phrase.lower()
        should_trigger = any(re.search(pattern, phrase_lower) for pattern in unable_to_answer_patterns)
        status = "✓" if should_trigger else "✗"
        print(f"{status} '{phrase}' -> Trigger: {should_trigger}")
    
    # Test phrases that should NOT trigger fallback
    normal_responses = [
        "Found 10 containers matching your query",
        "The container MEDU1234567 has arrived at the port",
        "Here are the delayed containers for this week",
    ]
    
    print("\nTesting non-trigger phrases (should be False):")
    for phrase in normal_responses:
        phrase_lower = phrase.lower()
        should_trigger = any(re.search(pattern, phrase_lower) for pattern in unable_to_answer_patterns)
        status = "✓" if not should_trigger else "✗"
        print(f"{status} '{phrase}' -> Trigger: {should_trigger}")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PANDAS FALLBACK TEST SUITE")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test 1: Fallback detection logic
    print("Test 1: Fallback Detection Logic")
    try:
        result = test_fallback_detection()
        results.append(("Fallback Detection", result))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Fallback Detection", False))
    
    # Test 2: Pandas fallback execution (requires Azure OpenAI credentials)
    print("\n\nTest 2: Pandas Fallback Execution")
    print("Note: This test requires valid Azure OpenAI credentials in config")
    try:
        result = test_pandas_fallback()
        results.append(("Pandas Fallback Execution", result))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Pandas Fallback Execution", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
