from datasets.process import process_jsonl
from llm.llm import HuggingfaceLLM
from rag.pure_rag import PureRag
from llm.eval import evaluate_model

def main():
    input_path = ""
    output_path = ""

    dataset = process_jsonl(input_path,output_path,topk=5)
    rag = PureRag(HuggingfaceLLM)

    evaluate_model(rag,dataset)