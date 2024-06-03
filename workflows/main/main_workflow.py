#workflows/main/main_workflow.py
import random
from typing import AsyncGenerator, List
from langgraph.graph import END, StateGraph
from langchain.schema.runnable import Runnable

from workflows.main.main_state import MainState

MAX_CHARACTERS = 1000

class MainWorkflow():
    """Class implementing the main workflow of the chatbot."""

    async def process_user_request(self, user_request: str) -> AsyncGenerator[str, None]:
        """
        This method processes the user request.
        """
        workflow = self._generate_workflow()
        initial_state = MainState(rejected=False, reason_rejected='', chat_history=user_request, language='', instructions=[])
        final_state = await workflow.ainvoke(initial_state)

        if final_state['rejected']:
            coroutine = self._bounce(final_state['reason_rejected'], final_state['language'])
        else:
            coroutine = self._responder(final_state['instructions'])
        async for chunk in coroutine:
            yield chunk
    
    def _generate_workflow(self) -> Runnable:
        """
        This method generates the graph and compile it into a workflow.
        """
        graph = StateGraph(MainState)
        graph.add_node('guard', self._guard)
        graph.add_node('think', self._think)
        graph.add_conditional_edges('guard', self._is_rejected, {'think': 'think', 'reject': END})
        graph.add_edge('think', END)
        graph.set_entry_point('guard')
        workflow: Runnable = graph.compile()
        return workflow

    # Private nodes and LCEL

    async def _guard(self, state: MainState) -> MainState:
        """ Guard node """
        rejected = random.choice([True, False])
        if len(state['chat_history']) > MAX_CHARACTERS:
            rejected = True
            reason_rejected = f'The user message is too long: {len(state["chat_history"])} characters, the maximum is {MAX_CHARACTERS}'
        else:
            if rejected:
                reason_rejected = 'Unlucky, try again'
            else:
                reason_rejected = 'You were lucky'
        return {'rejected': rejected, 'reason_rejected': reason_rejected}

    async def _think(self, _: MainState) -> MainState:
        """ TBD """
        return { 
            'instructions': [
                "Apologize to the user because the project isn't finished yet",
                "Ask the user to come back in a few days",
                "End the conversation"
            ],
        }
    
    async def _bounce(self, reason_rejected: str, _: str) -> AsyncGenerator[str, None]:
        """ Bounce LCEL"""
        for char in reason_rejected:
            yield char
    
    async def _responder(self, instructions: List[str]) -> AsyncGenerator[str, None]:
        """
        This method is a responder node that responds to the user input.
        """
        for instruction in instructions:
            yield instruction
            yield '\n'
        
    # Conditional edges

    async def _is_rejected(self, state: MainState) -> str:
        """ Conditional edge from the guard node """
        return 'reject' if state['rejected'] else 'think'

    # other methods
    def generate_ascii(self) -> str:
        """
        This method generates an ASCII representation of the workflow.
        """
        workflow = self._generate_workflow()
        return workflow.get_graph().draw_ascii()
