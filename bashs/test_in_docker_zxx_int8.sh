#!/bin/bash

#curl -X POST "http://localhost:8084/v1/generate" \
#    -H "Content-Type: application/json" \
#    -d '{
#        "text":["<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Hello, my name is <role_end><role>ASSISTANT</role>"],
#	"stream": true,
#	"sampling_params": {
#       	  "temperature": 0,
#          "max_new_tokens": 128
#	}
#    }'

#curl http://localhost:8081/v1/chat/completions \
#  -H "Content-Type: application/json" \
#  -H "Authorization: Bearer dummy_key" \
#  -d '{
#    "model": "Qwen3-32B-128k-concurrency1",
#    "messages": [
#      {"role": "system", "content": "你是一个可以调用工具的助手。"},
#      {"role": "user", "content": "请调用get_weather函数，查询北京的天气。/no_think"}
#    ],
#    "tools": [
#      {
#        "type": "function",
#        "function": {
#          "name": "get_weather",
#          "description": "获取指定城市的实时天气",
#          "parameters": {
#            "type": "object",
#            "properties": {
#              "city": {"type": "string", "description": "城市名"}
#            },
#            "required": ["city"]
#          }
#        }
#      }
#    ],
#    "tool_choice": "auto",
#    "max_tokens": 256,
#    "chat_template_kwargs": {"enable_thinking": true}
#  }'


#curl -sS http://localhost:8000/v1/chat/completions \
#  -H "Content-Type: application/json" \
#  -d '{
#      "model": "LLaDA2.0-mini",
#      "messages": [
#          {"role": "user", "content": "tell me something about huawei."}
#      ],
#      "temperature": 0.0,
#      "ignore_eos": false,
#      "max_tokens": 512
#  }'
#

#curl http://localhost:8000/v1/chat/completions \
#  -H "Content-Type: application/json" \
#  -d '{
#    "model": "/workspace/models/LLaDA/git_download/LLaDA2.0-mini-preview",
#    "messages": [
#      {"role": "user", "content": "你是谁？"}
#    ],
#    "temperature": 0.0,
#    "max_tokens": 1024,
#    "ignore_eos": false
#  }'


#curl http://localhost:8000/v1/chat/completions \
#  -H "Content-Type: application/json" \
#  -d '{
#    "model": "LLaDA2.0-mini",
#    "messages": [
#      {"role": "user", "content": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck."}
#    ],
#    "temperature": 0.0,
#    "max_tokens": 1024,
#    "ignore_eos": false
#  }'


# curl http://localhost:8000/v1/chat/completions \
#  -H "Content-Type: application/json" \
#  -d '{
#    "model": "LLaDA2.0-mini",
#    "messages": [
#      {"role": "user", "content": "Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?\nAnswer:"}
#    ],
#    "temperature": 0.0,
#    "max_tokens": 25600,
#    "ignore_eos": false
#  }'



curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LLaDA2.1-mini",
    "messages": [
      {"role": "user", "content": "(Question: Elizas rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? Answer:"}
    ],
    "temperature": 0.0,
    "max_tokens": 512,
    "ignore_eos": false
  }' \
  -w "\nTotal time: %{time_total}s\n"
