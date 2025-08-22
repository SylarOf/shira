from llm.prompts import PROMPTS


class PureRag:
    def __init__(self, llm):
        self.llm = llm 
        self.conversion = []
        self.conversion.append({"role": "system", "content":PROMPTS['system_pure_query']})
    
       

    
    def invoke(self,each_qa):
        ans = []

        question_prompt = "\nQuestion:\n" + each_qa['question']
        if question_prompt[-1] != '?':
            question_prompt += '?'
        
        files_prompt = "files:\n" + "\n".join(each_qa['query_text'])
        user_pure_query = "\n\n".join([files_prompt,question_prompt])


        self.conversion.append({"role":"user", "content":user_pure_query})

        output = self.llm(self.conversion)
        ans.append(output)

        self.conversion.append({"role":"assistant", "content":output})

        self.conversion.append({"role":"user", "content":PROMPTS['cot_query']})

        output = self.llm(self.conversion)

        ans.append(output)

        return ans


