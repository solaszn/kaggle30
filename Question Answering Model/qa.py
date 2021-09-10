from ir import Document, Query
# from IRQA.utils import progress_wrapped
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import QuestionAnsweringPipeline
from typing import Dict

class QuestionAnswering:
    """An easy interface for huggingface's Transformers pipeline. """

    # @progress_wrapped(estimated_time=36)
    def __init__(self, model_ckpt: str=None, custom_tokenizer: str=None) -> None:
        
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer)
        self.pipeline = QuestionAnsweringPipeline(model=self.model, tokenizer=self.tokenizer, \
                                        framework='pt')

    def answer(self, passage: Document, query: Query) -> Dict:
        result_set = {"book" : passage.book, "chapter" : passage.chapter_no, \
                        "context" : passage.text}
        result = self.pipeline(context= str(passage), question= query.text)
        result_set['answer'] = result['answer']
        result_set['score'] = result['score']
        return result_set