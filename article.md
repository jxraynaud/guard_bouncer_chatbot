# How to protect your chatbot using LangGraph

## Introduction

Every chatbot that isn't and open-ended chatbot, like ChatGPT or Claude, requires some kind of guard to prevent the user from asking questions that aren't strictly related to the intended usage of the chatbot.

The best way to do it is to implement a multi-agent system with an LLM used to decide if the user request is valid or not and a few trick to prevent chat injection. That's what we are going to try here.

As it's also a way to illustrate what can be done with LangGraph, we're going to use a step-by-step approach. I'll try to explain each step the best I can, underlying the reasoning rather than the code. However, the intended audience is supposed to be at least familiar with Python, the concept of a multi agents system and the basics of LLMs.

This article is the first of a serie so stay tuned !

## Specifications

### Functional

Such a system MUST be able to:
- Protect from prompt injection. The objective here is to make it hard, not to make it impossible. That's a general principle in security : you can't totally protect a system, you can only make it harder to break, however if it requires more resources to break the system than the value the user can get from it, then the system is kind of secure.
- Be able to reject toxic or off context prompts. 
- Be able to tell the user why the prompt was rejected, keep the chat history clean - we don't want to keep malignent prompts in the context - and be able to ask the user to rephrase its input if it was rejected.
- Go fast : we want the process to be as smooth and as fast as possible for the end user and that means as fast as possible. The chatbot will probably use agents in the background, that, by itself, might slow the system, so we want to make the guard as fast possible.
- We're going to include some placeholder to be able to add everything necessary to handle the user request when the guard didn't reject it.

We will also need to answer the following questions before:
- Which limit will we enforce regarding the user input length ? And in which unit ? Token is the easiest to deal with, but it's not the most user friendly. Number of words is more user friendly, but the rule of 4 tokens for 3 words is only true in English.
- How will we deal with the chat history ? The context is key for the guard to judge the last user input, so we need at least the 3 previous user messages and probably the last chatbot response. However more requests/responses mean more token to process, slower first token time, more expensive inference and potential exceed the llm's context size.

### Technical

To implement this, we're going to use the following stack:
- [Chainlit](https://docs.chainlit.io/get-started/overview) to build the chatbot. It's very simple and provides a lot of functionalities out of the box for very little code.
- [LangGraph](https://python.langchain.com/docs/langgraph/) to build the network of agents.
- [Langchain LCEL](https://python.langchain.com/docs/expression_language/get_started/) to call the llms. A case could be done to use DSPy instead to automate prompt optimization, but the code would be too complex.
- [Groq](https://wow.groq.com/) as the LLM API provider as it's probably the best inference speed we can't get and it's currently free. Note that the free tier is limited to 30RPM/15000 tokens per minute, so it isn't fit for production.
- [Llama 3 8B Instruct](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md). It's small, so it's fast, it's free, it's new and it's performant. The inference cost is very low, $0.05/$0.10 per million tokens. However, the context size is very small for 2024 standards, and it impacts the limits of user request and chat history defined above. As an alternative, Claude Haiku is also really fast, has a nice time to first token, a 200k token context size and is quite afordable at $0.25/$1.25 per million tokens.
- [Claude 3 sonnet](https://www.anthropic.com/news/claude-3-family) will also be used. It's reasonably fast and really good at redacting answers, without the robotic style of the GPT models, can follow instructions, and can speak multiple languages. As an alternative Llama 3 70B can be used, but only if you don't need more context size and you only need English. While much more expansive than the other models at $3/$15 per million tokens, it's still quite afordable for most usecases.

Last but not least, we will use classes, mostly because, like all old devs, I like object oriented programming.

### Agents

To implement the whole logic we will need the following agents or group of agents:
- A Guard : its role is to judge if the user input is toxic, unsafe (prompt injection) or off topic. It will use Llama 3 8B Instruct or Claude 3 haiku.
- A Memorizer : its role is to summarize the user input when the chat history becomes too large. We will use a simpler logic here, but it should be LLM-based in a real case scenario.
- A Bouncer : when the guard rejects the user input, the bouncer will tell the user why. It will use Llama 3 8B Instruct or Claude 3 haiku.
- A think workflow : that's usually where we like to implement the logic of the chatbot, whatever it does, a RAG or a multi agent system. 
- A Responder : It will use the provided instructions to respond to the user only if the guard didn't reject the user input. It will use Claude 3 sonnet.

## Implementation: Chainlit and LangGraph

### Step 1: Basic chainlit app

Our first objectif is to have a chainlit app that calls an object MainWorkflow that replies to the user by streaming back each character in the initial message. Then we will implement the graph that we will use later on.

The code for the app.py with the chainlit logic:

```python
#app.py
import chainlit as cl
from workflows.main.main_workflow import MainWorkflow

ARCHIVE_MIN_CALLS = 3

@cl.on_chat_start
async def on_chat_start():
    """
    This code is executed when the chat starts.
    """
    workflow = MainWorkflow()
    cl.user_session.set("workflow", workflow)

@cl.on_message
async def main(message: cl.Message):
    """
    main is a function that processes the user input and returns the response.
    """
    # Set the msg variable that will be sent back to the client and get sellbotix
    msg = cl.Message(content='')
    workflow: MainWorkflow = cl.user_session.get('workflow')

    # Stream the response
    async for chunk in workflow.process_user_reques(user_request=message.content):
        await msg.stream_token(chunk)
    await msg.send()
```

And the code for the workflow class:
    
```python
#workflows/main/main_workflow.py
from typing import AsyncGenerator


class MainWorkflow():
    
    async def process_user_request(self, user_request: str) -> AsyncGenerator[str, None]:
        """
        This method processes the user request.
        """
        for char in user_request:
            yield char
        
```

At this stage, you should be able to launch the app and see the characters of the user input being streamed back to you. Joking, without an await asyncio.sleep(0.1) in the MainWorkflow, it will be wait too fast for you to see anything.

### Step 2: Implement the graph

#### The state

To implement the graph, we first need to define a state, as LangGraph is a state machine for llm agents. We will also need some dummy functions for the nodes and the conditional edges of the graph.

In the state we need to store the following elements:
- **rejected**: a *boolean* that tells if the last user input was rejected or not by the guard. Afterall that's the whole objective of this article.
- **rejected_reason**: a *string* that tells the user why the last user input was rejected. We will need the bouncer to write an explanation for the user without providing him the initial request, we wouldn't want the bouncer to be prompt injected wouldn't we ?
- **language**: a *string* with the language used in the last user input. We will need to pass it to the bouncer as it will need to respond using the same language.
- **chat_history_string**: a *string* that contains the last n user inputs and the corresponding chatbot responses.
- **instructions**: a List of *string* that contains the instructions to be used by the responder.

Obviously, that's a simplified version, as, in a real case scenario, additional keys will be added to store the results of the "think workflow". For example, documentation links for a RAG, or a visitor profile.


```python
#worflows/main/main_state.py
from typing import TypedDict, Optional


class MainState(TypedDict):
    """State used by the LangGraph MainWorkflow."""
    rejected: bool
    reason_rejected: Optional[str]
    chat_history: str
    language: Optional[str]
    instructions: Optional[List[str]]
```

#### Dummy functions without the memorize logic

Using LangGraph is quite simple : you define a State. Then you define the node. They must be  functions (but please don't do that!) or coroutines - async functions - that takes a state as input and returns a state as output. Then you define edges. Some of them will be conditional edges and each one will use a function to implement the conditional logic.

So what do we need as coroutines and functions ?
- **guard**: get the state, call the LLM, return the state with the rejected, the reason_rejected and the language values updated.
- **check_guard**: read the state rejected key and END the workflow if it's True, continue to the Think node if it isn't.
- **think**: that's where we call the ThinkWorkflow, with all the application logic.

If you paid attention you might ask yourself why the Responder and the Bouncer aren't included. The reason is that it's cleaner to stream from a LCEL than from a LangGraph node, and it's a much better UX.

We also add a _generate_workflow method that will be called by the process_user_request method.

```python
#workflows/main/main_workflow.py
import random
from typing import AsyncGenerator, List
from langgraph.graph import END, StateGraph
from langchain.schema.runnable import Runnable

from workflows.main.main_state import MainState


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

    async def _guard(self, _: MainState) -> MainState:
        """ Guard node """
        rejected = random.choice([True, False])
        return {'rejected': rejected, 'reason_rejected': 'Unlucky, try again' if rejected else 'You were lucky'}

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
```

At this state we have a working implementation, yet not really useful, but it allows us to test the logic of the graph.

### Step 3: Get a nice Ascii of the graph.

Langgraph provides an easy way to get an ascii representation of the graph. So we will just create a generate_ascii method that wil be call from a generate_ascii_graph.py file.

```python
#workflows/main/main_workflow.py
class MainWorkflow():
    ... # GPT4 lazy version mode activated ;)
    # other methods
    def generate_ascii(self) -> str:
        """
        This method generates an ASCII representation of the workflow.
        """
        workflow = self._generate_workflow()
        return workflow.get_graph().draw_ascii()
```

```python
#generate_ascii_graph.py
from workflows.main.main_workflow import MainWorkflow


def main():
    workflow = MainWorkflow()
    print(workflow.generate_ascii())

if __name__ == '__main__':
    main()
```

And a `python generate_ascii_graph.py` should print this:

```
        +-----------+    
        | __start__ |
        +-----------+
              *
              *
              *
          +-------+
          | guard |
          +-------+
              *
              *
              *
      +--------------+
      | _is_rejected |
      +--------------+
         *         **
       **            *
      *               **
+-------+               *
| think |             **
+-------+            *
         *         **
          **     **
            *   *
        +---------+
        | __end__ |
        +---------+
```

## Conclusion

So at this stage, we have something that is working, yet completely useless. Adding the LLM logic will be necessary, and there's a lot to talk about, but first, before sending anything to an LLM, we will have to add the Memorize logic to the graph and a few more things to the Class.
