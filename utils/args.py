from dotenv import load_dotenv
import os
import argparse

# load .env file info
load_dotenv()

# API Keys
def return_API_keys():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_1": os.getenv("DEEPSEEK_API_KEY_1", "your_default_api_key_here"),
    }

# argparse
# https://stackoverflow.com/questions/46719811/best-practices-for-writing-argparse-parsers
def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
    parser.add_argument('--graph-state-path', type=str, default='', help='Path to save graph state')
    parser.add_argument('--molecule_type', type=str, default='chromophore', help='Type of molecule to generate')
    parser.add_argument('--project-name', type=str, default='guacamol_logP', help='wandb project name')
    parser.add_argument('--topk', type=int, default=3, help='TOP k molecules')
    parser.add_argument('--task', type=str, default='albuterol_similarity', help='List of tasks to run')#,nargs='+', ) 
    parser.add_argument('--property-name', type=str, default='logP', help='Conditional property')
    parser.add_argument('--property-value', type=float, default=2.0, help='Conditional property value')
    parser.add_argument('--property-unit', type=str, default='', help='Unit of conditional property')
    parser.add_argument('--property-threshold', type=int, default=0, help='Molweight threshold parameter')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--api-num', type=int, default=1, help='API number for molecule generation')
    parser.add_argument('--scientist-temperature', type=float, default=1.5, help='Temperature setting for scientist LLM')
    parser.add_argument('--reviewer-temperature', type=float, default=1.5, help='Temperature setting for reviewer LLM')
    parser.add_argument('--double-checker-temperature', type=float, default=1.5, help='Temperature setting for reviewer LLM')
    parser.add_argument('--tool-call-temperature', type=float, default=1.5, help='Temperature setting for tool call LLM')
    parser.add_argument('--scientist-model-name', type=str, default="deepseek-chat", help='Scientist LLM model, gpt-4o')
    parser.add_argument('--reviewer-model-name', type=str, default="deepseek-chat", help='Reviewer LLM model, gpt-4o')
    parser.add_argument('--double-checker-model-name', type=str, default="deepseek-chat", help='Double checker LLM model, gpt-4o')
    parser.add_argument('--tool-call-model-name', type=str, default="deepseek-chat", help='Tool call LLM model, gpt-4o')
    parser.add_argument('--doc-batch-size', type=int, default=50, help='Batch size for document processing')
    
    return parser.parse_args(*args)