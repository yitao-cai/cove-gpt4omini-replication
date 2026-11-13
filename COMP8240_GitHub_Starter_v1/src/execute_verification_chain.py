"""Execute self-verification for CoVe, compatible with LangChain 1.x.

Closed-book by default. Individual verification prompts do not include
baseline_response. Original per-question generate_prompt behavior preserved.
"""
import itertools
import re
from typing import Any, Dict, List, Optional

from langchain_classic.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from pydantic import ConfigDict

import prompts


class ExecuteVerificationChain(Chain):
    """Execute each generated verification question independently."""

    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    input_key: str = "verification_questions"
    output_key: str = "verification_answers"
    use_search_tool: bool = False
    search_tool: Any = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def search_for_verification_question(self, verification_question: str) -> str:
        if self.search_tool is None:
            raise RuntimeError("Search is not configured. Closed-book experiments must keep use_search_tool=False.")
        return self.search_tool.run(verification_question)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        sub_inputs = {k: v for k, v in inputs.items() if k == self.input_key}
        verification_questions_str = self.prompt.format_prompt(**sub_inputs).to_string()
        verification_questions_list = [
            re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            for line in verification_questions_str.splitlines()
            if line.strip()
        ]
        if not verification_questions_list:
            raise ValueError("No verification questions were generated.")

        execution_prompt_search_tool = PromptTemplate.from_template(
            prompts.EXECUTE_PLAN_PROMPT_SEARCH_TOOL
        )
        execution_prompt_self_llm = PromptTemplate.from_template(
            prompts.EXECUTE_PLAN_PROMPT_SELF_LLM
        )
        verification_answers_list = []
        for question in verification_questions_list:
            if self.use_search_tool:
                search_result = self.search_for_verification_question(question)
                execution_prompt_value = execution_prompt_search_tool.format_prompt(
                    search_result=search_result, verification_question=question
                )
            else:
                execution_prompt_value = execution_prompt_self_llm.format_prompt(
                    verification_question=question
                )
            result = self.llm.generate_prompt(
                [execution_prompt_value],
                callbacks=run_manager.get_child() if run_manager else None,
            )
            verification_answers_list.append(result.generations[0][0].text)

        pairs = "".join(
            "Question: {} Answer: {}\n".format(question, answer)
            for question, answer in itertools.zip_longest(
                verification_questions_list, verification_answers_list
            )
        )
        return {self.output_key: pairs}

    # For async calls, inherit Chain._acall, which runs the above sync _call.

    @property
    def _chain_type(self) -> str:
        return "execute_verification_chain"
