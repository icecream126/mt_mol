from dotenv import load_dotenv
import os
import argparse

# load .env file info
load_dotenv()

# API Keys
def return_API_keys():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your_default_api_key_here"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_YUNHUI": os.getenv("OPENAI_API_KEY_YUNHUI", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_1": os.getenv("DEEPSEEK_API_KEY_1", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_2": os.getenv("DEEPSEEK_API_KEY_2", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_3": os.getenv("DEEPSEEK_API_KEY_3", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_4": os.getenv("DEEPSEEK_API_KEY_4", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_5": os.getenv("DEEPSEEK_API_KEY_5", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_6": os.getenv("DEEPSEEK_API_KEY_6", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_7": os.getenv("DEEPSEEK_API_KEY_7", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_8": os.getenv("DEEPSEEK_API_KEY_8", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_9": os.getenv("DEEPSEEK_API_KEY_9", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_10": os.getenv("DEEPSEEK_API_KEY_10", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_11": os.getenv("DEEPSEEK_API_KEY_11", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_12": os.getenv("DEEPSEEK_API_KEY_12", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_13": os.getenv("DEEPSEEK_API_KEY_13", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_14": os.getenv("DEEPSEEK_API_KEY_14", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_15": os.getenv("DEEPSEEK_API_KEY_15", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_16": os.getenv("DEEPSEEK_API_KEY_16", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_17": os.getenv("DEEPSEEK_API_KEY_17", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_18": os.getenv("DEEPSEEK_API_KEY_18", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_19": os.getenv("DEEPSEEK_API_KEY_19", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_20": os.getenv("DEEPSEEK_API_KEY_20", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_21": os.getenv("DEEPSEEK_API_KEY_21", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_22": os.getenv("DEEPSEEK_API_KEY_22", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_23": os.getenv("DEEPSEEK_API_KEY_23", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_24": os.getenv("DEEPSEEK_API_KEY_24", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_25": os.getenv("DEEPSEEK_API_KEY_25", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_26": os.getenv("DEEPSEEK_API_KEY_26", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_27": os.getenv("DEEPSEEK_API_KEY_27", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_28": os.getenv("DEEPSEEK_API_KEY_28", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_29": os.getenv("DEEPSEEK_API_KEY_29", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_30": os.getenv("DEEPSEEK_API_KEY_30", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_31": os.getenv("DEEPSEEK_API_KEY_31", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_32": os.getenv("DEEPSEEK_API_KEY_32", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_33": os.getenv("DEEPSEEK_API_KEY_33", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_34": os.getenv("DEEPSEEK_API_KEY_34", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_35": os.getenv("DEEPSEEK_API_KEY_35", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_36": os.getenv("DEEPSEEK_API_KEY_36", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_37": os.getenv("DEEPSEEK_API_KEY_37", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_38": os.getenv("DEEPSEEK_API_KEY_38", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_39": os.getenv("DEEPSEEK_API_KEY_39", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_40": os.getenv("DEEPSEEK_API_KEY_40", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_41": os.getenv("DEEPSEEK_API_KEY_41", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_42": os.getenv("DEEPSEEK_API_KEY_42", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_43": os.getenv("DEEPSEEK_API_KEY_43", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_44": os.getenv("DEEPSEEK_API_KEY_44", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_45": os.getenv("DEEPSEEK_API_KEY_45", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_46": os.getenv("DEEPSEEK_API_KEY_46", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_47": os.getenv("DEEPSEEK_API_KEY_47", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_48": os.getenv("DEEPSEEK_API_KEY_48", "your_default_api_key_here"),
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