from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage



class LLMCaller:
    def __init__(self, api_key: str, model: str, system_prompt:str, identifier: str="None", verbose:bool = False):
        self.model = model
        self._api_key = api_key
        self.system_prompt = system_prompt
        self.identifier = identifier
        self.verbose = verbose


        self._initialize_LLM()

    def _initialize_LLM(self):
        self.llm = ChatGroq(
            model=self.model,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self._api_key,
        )

        print(
            f"Initialized LLMCaller with model: {self.model}"
        )

    def call(self, **kwargs: dict):
        prompt = self.system_prompt
        prompt = prompt.format(**kwargs)
        
        if self.verbose:
            print(f"___LOGS FROM {self.identifier} LLM___")
            print("PROMPT: ", prompt)
        
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        output = generated_text.content.strip()
        
        if self.verbose:
            print(f"OUTPUT OF {self.identifier} LLM:")
            print(output)
            print(f"___END OF LOGS FROM {self.identifier} LLM___")

        return output
