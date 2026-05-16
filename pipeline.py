from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1: Search
    print("\n" + " ="*50)
    print(f"Step 1 - Search Agent: {topic}")
    print("="*50)
    search_agent = build_search_agent()
    try:
        search_result = search_agent.invoke({"input": f"Research details on {topic}"})
        state["search_results"] = search_result['output']
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return {}

    # Step 2: Read/Scrape
    print("\n" + " ="*50)
    print("Step 2 - Reader Agent is working...")
    print("="*50)
    reader_agent = build_reader_agent()
    try:
        reader_result = reader_agent.invoke({
            "input": f"Look at these results: {state['search_results'][:1000]}. Pick one URL and scrape it."
        })
        state['scraped_content'] = reader_result['output']
    except Exception as e:
        print(f"❌ Reader Error: {e}")
        state['scraped_content'] = "No scraped content available."

    # Step 3: Write
    print("\n" + " ="*50)
    print("Step 3 - Writing Report...")
    print("="*50)
    research_combined = f"Search: {state['search_results']}\n\nScraped: {state['scraped_content']}"
    state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
    print(f"\nDraft:\n{state['report']}")

    # Step 4: Critique
    print("\n" + " ="*50)
    print("Step 4 - Reviewing...")
    print("="*50)
    state["feedback"] = critic_chain.invoke({"report": state['report']})
    print(f"\nFeedback:\n{state['feedback']}")

    return state

if __name__ == "__main__":
    t = input("\nEnter research topic: ")
    if t.strip():
        run_research_pipeline(t)
    else:
        print("Please enter a valid topic.")