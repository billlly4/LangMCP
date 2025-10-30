from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import List, Union, Dict, Any, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent # Still using this for the tool-calling logic
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Define the State for our graph
# messages: The conversation history
# next_step: Used for conditional routing
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_step: Literal["tool_use", "conversational", "handle_mixed", "__end__"] # To guide transitions

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": ["weather.py"],
                "transport": "stdio",
            }
        }
    )

    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    tools = await client.get_tools()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    # We'll still use create_react_agent for its robust tool-calling mechanism
    # but we'll invoke it selectively within our graph.
    tool_agent_runnable = create_react_agent(model, tools)

    # --- Nodes Definition ---

    # Node 1: Decide what to do based on user input (Router)
    async def route_question(state: AgentState) -> AgentState:
        print("\n--- Route Question Node ---", flush=True)
        # Use a simple LLM call to classify the intent
        # This LLM call needs access to the current message history
        last_message = state["messages"][-1]
        
        # We need a prompt for the router LLM to classify the input
        router_prompt = f"""You are an expert at routing user questions.
        Your goal is to classify the user's intent based on their last message.
        
        The categories are:
        - 'tool_use': The user explicitly asks to use a tool (e.g., "add 2 to 3", "what is the weather in LA", "multiply 5 by 7", "get alerts for CA").
        - 'conversational': The user asks a general question, makes a statement, tells a joke, expresses emotion, or wants to chat (e.g., "hello", "tell me a joke", "how are you", "what is my name", "I am sad").
        - 'handle_mixed': The user asks for both a tool action AND a conversational response in the same message (e.g., "tell me a joke and then add 2 to 3", "I'm having a hard time, add 5 to 10").

        Return only one of the following words: 'tool_use', 'conversational', 'handle_mixed'.
        
        User message: {last_message.content}
        """
        
        response = await model.ainvoke(router_prompt)
        decision = response.content.strip().lower()

        print(f"Router Decision: {decision}", flush=True)

        if "tool_use" in decision:
            return {"next_step": "tool_use", "messages": state["messages"]}
        elif "conversational" in decision:
            return {"next_step": "conversational", "messages": state["messages"]}
        elif "handle_mixed" in decision:
            return {"next_step": "handle_mixed", "messages": state["messages"]}
        else:
            # Default to conversational if classification is unclear
            print(f"Warning: Router returned unclear decision '{decision}', defaulting to conversational.", flush=True)
            return {"next_step": "conversational", "messages": state["messages"]}

    # Node 2: Handle tool-using requests using the prebuilt agent
    async def call_tool_node(state: AgentState) -> AgentState:
        print("\n--- Call Tool Node ---", flush=True)
        # Invoke the prebuilt agent which is designed to use tools
        try:
            response_from_agent: Union[BaseMessage, Dict[str, Any]] = await tool_agent_runnable.ainvoke({"messages": state["messages"]})
            
            ai_response_message: BaseMessage = None
            
            if isinstance(response_from_agent, BaseMessage):
                ai_response_message = response_from_agent
            elif isinstance(response_from_agent, dict):
                if "messages" in response_from_agent and isinstance(response_from_agent["messages"], list):
                    last_message = response_from_agent["messages"][-1]
                    if isinstance(last_message, BaseMessage):
                        ai_response_message = last_message
            
            if ai_response_message:
                return {"messages": state["messages"] + [ai_response_message], "next_step": "__end__"}
            else:
                # Fallback if no valid message could be extracted from agent
                error_msg = AIMessage(content=f"Sorry, I had trouble processing your tool request. Raw agent response: {response_from_agent}")
                return {"messages": state["messages"] + [error_msg], "next_step": "__end__"}

        except Exception as e:
            error_msg = AIMessage(content=f"An error occurred while using a tool: {e}. Please try again.")
            return {"messages": state["messages"] + [error_msg], "next_step": "__end__"}

    # Node 3: Handle purely conversational requests
    async def generate_conversational_response(state: AgentState) -> AgentState:
        print("\n--- Conversational Response Node ---", flush=True)
        # Use the LLM directly for a conversational response
        try:
            response = await model.ainvoke(state["messages"])
            return {"messages": state["messages"] + [response], "next_step": "__end__"}
        except Exception as e:
            error_msg = AIMessage(content=f"An error occurred during conversation: {e}. Please try again.")
            return {"messages": state["messages"] + [error_msg], "next_step": "__end__"}

    # Node 4: Handle mixed requests (more complex, for now we'll prioritize tool if available)
    async def handle_mixed_request_node(state: AgentState) -> AgentState:
        print("\n--- Handle Mixed Request Node ---", flush=True)
        last_message_content = state["messages"][-1].content
        
        # Strategy for mixed: Try to extract the tool-related part and the conversational part.
        # This is a hard problem for LLMs, but we can instruct it.
        # For simplicity, let's first try to perform the tool action.
        # Then, we'll try to add a conversational part.

        # Step 1: Attempt tool usage with the whole prompt (the prebuilt agent is good at this)
        # The tool_agent_runnable is designed to identify and execute tools.
        tool_response_state = await call_tool_node(state) # This will execute the tool if found
        
        # Check if a tool was actually used and a result obtained.
        # A simple heuristic: if the last message in tool_response_state is from a ToolMessage
        # or if it's an AIMessage that looks like a tool result.
        
        # Let's get the latest messages *after* tool attempt
        updated_messages = tool_response_state["messages"]

        final_ai_message = updated_messages[-1]
        
        # If the last response was a tool result, or an LLM response to a tool call
        if final_ai_message and "An error occurred" not in final_ai_message.content:
            # Now, try to add a conversational aspect if the original message had one.
            # This requires another LLM call to synthesize a response.
            
            # Create a new prompt to the LLM to provide a conversational response
            # considering the original user query and the tool result.
            synthesize_prompt = [
                *state["messages"], # Original messages up to user's mixed query
                final_ai_message, # The result from the tool agent
                HumanMessage(content=f"The user also asked for a conversational response (like a joke or comforting message) in their original query: '{last_message_content}'. Considering the tool result you just provided, please also address the conversational part of their request.")
            ]
            
            try:
                conversational_addition = await model.ainvoke(synthesize_prompt)
                # Combine the tool result message and the conversational message
                combined_response = AIMessage(
                    content=f"{final_ai_message.content}\n\n{conversational_addition.content}"
                )
                return {"messages": state["messages"] + [combined_response], "next_step": "__end__"}
            except Exception as e:
                print(f"Warning: Could not synthesize conversational part for mixed request: {e}", flush=True)
                # If synthesis fails, just return the tool result
                return {"messages": updated_messages, "next_step": "__end__"}
        else:
            # If the tool attempt failed or didn't produce a clear result,
            # fall back to a conversational response.
            print("Mixed request: Tool execution failed or wasn't clear, falling back to conversational.", flush=True)
            return await generate_conversational_response(state)

    # --- Graph Definition ---

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route_question", route_question)
    graph.add_node("call_tool_node", call_tool_node)
    graph.add_node("generate_conversational_response", generate_conversational_response)
    graph.add_node("handle_mixed_request_node", handle_mixed_request_node)

    # Define the entry point
    graph.set_entry_point("route_question")

    # Define conditional edges
    graph.add_conditional_edges(
        "route_question",  # From this node
        lambda state: state["next_step"], # Based on the 'next_step' in the state
        {
            "tool_use": "call_tool_node",
            "conversational": "generate_conversational_response",
            "handle_mixed": "handle_mixed_request_node",
        }
    )

    # All final response nodes transition to END to complete the current turn
    graph.add_edge("call_tool_node", END)
    graph.add_edge("generate_conversational_response", END)
    graph.add_edge("handle_mixed_request_node", END)

    # Compile the graph
    app = graph.compile()

    print("AI Agent initialized. Type 'exit' to quit.")
    
    messages: List[BaseMessage] = [] 

    user_input = ""
    while user_input.lower() != "exit":
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        user_message = HumanMessage(content=user_input)
        messages.append(user_message)
        
        try:
            # Pass the current messages as the initial state for the graph
            # The graph will update the messages list as it progresses
            final_state = await app.ainvoke({"messages": messages})
            
            # The 'messages' list in the final state will contain the full history
            # including the AI's final response for this turn.
            messages = final_state["messages"] # Update our external messages list
            
            ai_response = messages[-1] # The last message is the AI's response

            # We need to print the AI's response content
            # The previous robust extraction logic is now inside the nodes,
            # so here we just print the last message's content directly.
            print(f"AI: {ai_response.content}")

        except Exception as e:
            print(f"An error occurred during agent invocation: {e}")
            # If an error occurs, remove the last user message to avoid
            # passing a failed turn to the next invocation.
            if messages and messages[-1] == user_message:
                messages.pop() 
            print("Please try again.")

asyncio.run(main())