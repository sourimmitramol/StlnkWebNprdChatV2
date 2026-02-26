# Pandas Code Generation Fallback

## Overview
The chatbot now includes an intelligent pandas code generation fallback mechanism that automatically generates and executes pandas code when the predefined tools cannot handle a user's query.

## How It Works

### 1. Primary Agent Execution
When a user asks a question, the system first attempts to route it through:
- **Predefined Tools**: Specialized functions for common queries (containers, POs, delays, tracking, etc.)
- **Router Logic**: Pattern matching to find the most suitable tool

### 2. Fallback Trigger Conditions
The pandas fallback is automatically triggered when:

1. **Agent couldn't understand**: Response contains phrases like:
   - "I couldn't understand"
   - "I don't have enough information"
   - "Please try rephrasing"
   - "Unable to answer"
   - "I don't know"

2. **No tools were called**: The agent runs but doesn't invoke any tool (empty intermediate_steps)

3. **Router returned error**: Query router returns "I couldn't understand your query"

### 3. Pandas Code Generation
When triggered, the fallback:
1. **Retrieves DataFrame schema**: Gets column names, data types, and sample values
2. **Generates pandas code**: Uses Azure OpenAI to generate Python pandas code
3. **Executes safely**: Runs the generated code in a controlled environment
4. **Formats results**: Presents the output in a user-friendly format

### 4. Safety Features
- **Restricted execution environment**: Only allows pandas, datetime, re, numpy imports
- **No dangerous operations**: Uses restricted `exec()` with limited builtins
- **Error handling**: Catches and logs any execution errors
- **Row limits**: Automatically limits output to 20 rows for display
- **Consignee filtering**: Respects access control based on user permissions

## Example Queries That Use Fallback

### Complex Analytical Queries
```
"What's the correlation between delay days and carrier performance?"
"Show me a pivot table of containers by port and month"
"Calculate the standard deviation of transit times"
```

### Custom Aggregations
```
"Group containers by final destination and count them"
"What's the median ETA across all carriers?"
"Show unique load ports with more than 100 containers"
```

### Advanced Filtering
```
"Find containers where revised ETA is more than 2 weeks after original ETA"
"Show me containers with null delivery dates but non-null ATA"
"List all POs with more than 5 containers"
```

## Code Example

Generated code might look like:
```python
# Find containers with longest transit times
df['eta_dp'] = pd.to_datetime(df['eta_dp'], errors='coerce')
df['atd_lp'] = pd.to_datetime(df['atd_lp'], errors='coerce')
df['transit_days'] = (df['eta_dp'] - df['atd_lp']).dt.days
result = df.nlargest(20, 'transit_days')[['container_number', 'transit_days', 'carrier', 'route']].head(20)
```

## Configuration

The fallback is configured in:
- `agents/azure_agent.py` - `pandas_code_generator_fallback()` function
- `api/main.py` - Fallback detection and invocation logic

### Key Parameters
- **Temperature**: 0.0 (deterministic code generation)
- **Max rows**: 20 (display limit)
- **Column limit**: 50 (for schema in prompt)
- **Timeout**: 30 seconds (execution limit)

## Monitoring

The fallback logs its activity:
```
[PANDAS FALLBACK] Triggered: Agent couldn't understand query
[PANDAS FALLBACK] Generating code for query: [user query]
[PANDAS FALLBACK] Generated code: [pandas code]
[PANDAS FALLBACK] Successfully generated response using pandas
```

Check logs at: `logs/shipping_chatbot.log`

## Response Format

When fallback is used, the API returns:
```json
{
  "response": "Found 15 record(s):\n\n[formatted results]",
  "observation": "Pandas code generation fallback",
  "table": [],
  "mode": "pandas-fallback",
  "session_id": "..."
}
```

## Limitations

1. **No external data sources**: Only works with loaded DataFrame
2. **Computation limits**: Complex calculations may timeout
3. **No file I/O**: Cannot read/write external files
4. **Memory constraints**: Large result sets are automatically limited

## Best Practices

### For Users
- Be specific about what you want to analyze
- Mention column names if you know them
- Specify time ranges clearly
- Ask one question at a time

### For Developers
- Review generated code in logs for quality
- Add frequently used patterns as dedicated tools
- Monitor execution times and errors
- Update schema information if columns change

## Troubleshooting

### Fallback not triggering
- Ensure query truly doesn't match any existing tools
- Check if router is returning proper error messages
- Verify intermediate_steps logic in main.py

### Code generation errors
- Review DataFrame schema sent to LLM
- Check if column names are being recognized
- Verify Azure OpenAI endpoint is responsive

### Execution failures
- Check for syntax errors in generated code
- Verify all required columns exist
- Look for type conversion issues with dates

## Future Enhancements

Potential improvements:
1. **Code caching**: Store successful code patterns for reuse
2. **Learning mechanism**: Improve code generation based on successful patterns
3. **Multi-step analysis**: Break complex queries into multiple pandas operations
4. **Data visualization**: Generate charts/graphs for analytical queries
5. **Performance optimization**: Index frequently queried columns

## Security Considerations

- Executes in restricted environment (no file access, network, etc.)
- Limited to pandas operations only
- Respects consignee access control
- All code generation is logged for audit
- No user input is directly executed (LLM generates code)
