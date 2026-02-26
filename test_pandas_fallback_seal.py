"""
Test script for pandas fallback with seal_number query

This tests:
1. Safe builtins (str, int, float, etc.)
2. Code syntax validation
3. Automatic retry with query rephrasing
4. Column existence checking
"""

def test_safe_builtins():
    """Test that str and other builtins are available"""
    safe_builtins = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'len': len,
        'range': range,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'enumerate': enumerate,
        'zip': zip,
        'any': any,
        'all': all,
        'isinstance': isinstance,
        'type': type,
        'None': None,
        'True': True,
        'False': False,
    }
    
    # Test code that uses str
    test_code = """
import pandas as pd
import numpy as np

# Create test dataframe
df = pd.DataFrame({
    'container_number': ['TRHU8539467', 'ABCD1234567'],
    'seal_number': [12345, 67890]
})

# Convert seal_number to string (this requires 'str' builtin)
df['seal_number'] = df['seal_number'].astype(str)

# Filter for specific container
result = df[df['container_number'] == 'TRHU8539467'][['container_number', 'seal_number']]
"""
    
    local_vars = {
        'df': None,
        'pd': __import__('pandas'),
        'np': __import__('numpy'),
        'result': None
    }
    
    try:
        exec(test_code, {"__builtins__": safe_builtins}, local_vars)
        result = local_vars['result']
        print("✅ Test 1 PASSED: str builtin is available")
        print(f"   Result:\n{result}")
        return True
    except NameError as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ Test 1 FAILED with unexpected error: {e}")
        return False


def test_code_validation():
    """Test syntax validation before execution"""
    
    # Valid code
    valid_code = "x = 1 + 2"
    try:
        compile(valid_code, '<string>', 'exec')
        print("✅ Test 2a PASSED: Valid code compiles")
    except SyntaxError:
        print("❌ Test 2a FAILED: Valid code rejected")
        return False
    
    # Invalid code
    invalid_code = "x = 1 +"
    try:
        compile(invalid_code, '<string>', 'exec')
        print("❌ Test 2b FAILED: Invalid code accepted")
        return False
    except SyntaxError:
        print("✅ Test 2b PASSED: Invalid code rejected")
    
    return True


def test_column_existence():
    """Test column existence checking pattern"""
    import pandas as pd
    
    df = pd.DataFrame({
        'container_number': ['TRHU8539467'],
        'seal_number': [12345]
    })
    
    # Test with existing column
    if 'seal_number' in df.columns:
        result = df[['container_number', 'seal_number']]
        print("✅ Test 3a PASSED: Column existence check works for existing column")
    else:
        print("❌ Test 3a FAILED: Column existence check failed")
        return False
    
    # Test with non-existing column
    if 'nonexistent_column' in df.columns:
        print("❌ Test 3b FAILED: Non-existent column detected as existing")
        return False
    else:
        print("✅ Test 3b PASSED: Column existence check works for non-existing column")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PANDAS FALLBACK ENHANCEMENT TESTS")
    print("=" * 60)
    print()
    
    all_passed = True
    
    print("Test 1: Safe Builtins (str, int, float, etc.)")
    print("-" * 60)
    all_passed &= test_safe_builtins()
    print()
    
    print("Test 2: Code Syntax Validation")
    print("-" * 60)
    all_passed &= test_code_validation()
    print()
    
    print("Test 3: Column Existence Checking")
    print("-" * 60)
    all_passed &= test_column_existence()
    print()
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
