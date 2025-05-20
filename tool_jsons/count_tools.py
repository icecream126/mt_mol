import os

# Get all tool jsons
tool_jsons = [f for f in os.listdir() if f.endswith('.json')]

# Count the number of tools
import json
sum=0
for tool_json in tool_jsons:
    with open(tool_json, 'r') as f:
        data = json.load(f)
    print('Tool: ', tool_json, 'Number of tools: ', len(data))
    # import pdb; pdb.set_trace()
    sum += len(data)

print(sum)