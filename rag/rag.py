from llm.prompts import PROMPTS

class Rag:
    def __init__(self,llm):
        self.llm = llm
        self.conversion = []
        self.conversion.append({"role":"system", "content":PROMPTS['sys_query']})

    def invoke(self,each_qa):
        res = []

        # in context learning
        self.conversion.append({"role":"user", "content":PROMPTS['icl_user_prompts']})
        self.conversion.append({"role":"user", "content":PROMPTS['icl_ass_prompt']})

        question_prompt = "\nQuestion:\n" + each_qa['question']
        if question_prompt[-1] != '?':
            question_prompt += '?'
        
        triplets_prompt = "Triplet:\n" + "\n".join(each_qa['triplets'])
        user_query = "\n\n".join([triplets_prompt,question_prompt])


        self.conversion.append({"role":"user", "content":user_query})
        output = self.llm(self.conversion)
        res.append(output)

        self.conversion.append({"role":"assistant","content":PROMPTS['cot_query']})
        output = self.llm(self.conversion)

        res.append(output)

        return res