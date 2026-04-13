"""Agentic RAG — Unified CLI Entry Point.

Usage:
    python app.py ingest <path>             Ingest documents into the knowledge base
    python app.py ask "<question>"          Ask a single question (basic RAG)
    python app.py chat                      Interactive Q&A loop (basic RAG)
    python app.py agent "<question>"        Ask using the agentic pipeline
    python app.py agent-chat                Interactive agentic chat with memory
    python app.py eval                      Run evaluation metrics
    python app.py diagnose                  Run pipeline diagnostics
"""
import sys
import os
import argparse

# Fix Windows console encoding for unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))


def cmd_ingest(args):
    """Ingest documents into the vector store."""
    from src.pipeline.ingest_pipeline import run_ingestion

    paths = args.paths
    print(f"\n📄 Ingesting documents from: {paths}")
    result = run_ingestion(paths)

    print(f"\n✅ Ingestion Complete")
    print(f"   Documents loaded : {result['documents_loaded']}")
    print(f"   Chunks created   : {result['chunks_created']}")
    print(f"   Chunks stored    : {result['chunks_stored']}")


def cmd_ask(args):
    """Ask a single question using basic RAG pipeline."""
    from src.pipeline.qa_pipeline import ask

    question = args.question
    print(f"\n❓ Question: {question}")
    print("   Searching and generating answer...\n")

    result = ask(question)

    print(f"💬 Answer:\n{result.answer}\n")
    if result.citations:
        print("📎 Sources:")
        for c in result.citations:
            print(f"   - {c['source']} (score: {c['score']:.3f})")


def cmd_chat(args):
    """Interactive Q&A loop using basic RAG pipeline."""
    from src.pipeline.qa_pipeline import ask

    print("\n💬 RAG Chat (basic pipeline)")
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        try:
            result = ask(question)
            print(f"\n🤖 Answer:\n{result.answer}\n")
            if result.citations:
                print("   Sources:", ", ".join(c["source"] for c in result.citations))
            print()
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def cmd_agent(args):
    """Ask a single question using the agentic pipeline."""
    from src.pipeline.agentic_pipeline import agent_ask

    question = args.question
    print(f"\n🤖 Agent Question: {question}")
    print("   Reasoning and using tools...\n")

    result = agent_ask(question)

    # Show reasoning trace
    if result.reasoning_steps:
        print("🧠 Reasoning Trace:")
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"\n   Step {i}:")
            if step.thought:
                print(f"   💭 Thought: {step.thought[:200]}")
            if step.action:
                print(f"   🔧 Action: {step.action}")
                print(f"   📥 Input: {step.action_input[:100]}")
            if step.observation:
                print(f"   👁️  Observation: {step.observation[:150]}...")

    print(f"\n💬 Final Answer:\n{result.final_answer}")
    print(f"\n📊 Stats: {result.total_llm_calls} LLM calls, tools used: {result.tools_used}")


def cmd_agent_chat(args):
    """Interactive agentic chat with memory."""
    from src.pipeline.agentic_pipeline import agent_chat, agent_reset

    print("\n🤖 Agentic RAG Chat (with memory & tools)")
    print("   Type your question and press Enter.")
    print("   Type 'reset' to clear memory.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break
        if question.lower() == "reset":
            agent_reset()
            print("🔄 Memory cleared.\n")
            continue

        try:
            result = agent_chat(question)

            # Show brief reasoning
            if result.reasoning_steps:
                tools = [s.action for s in result.reasoning_steps if s.action]
                if tools:
                    print(f"   🔧 Used: {', '.join(tools)}")

            print(f"\n🤖 {result.final_answer}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def cmd_eval(args):
    """Run evaluation metrics."""
    from src.evaluation.evaluator import run_evaluation

    dataset_path = args.dataset if hasattr(args, "dataset") and args.dataset else None
    print("\n📊 Running evaluation...\n")

    summary = run_evaluation(dataset_path=dataset_path)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  Questions evaluated : {summary['total_questions']}")
    print(f"  Successful          : {summary['successful']}")
    print(f"  Avg Recall@k        : {summary['avg_recall_at_k']:.3f}")
    print(f"  Avg Faithfulness    : {summary['avg_faithfulness']:.3f}")
    print(f"  Avg Relevance       : {summary['avg_answer_relevance']:.3f}")


def cmd_diagnose(args):
    """Run pipeline diagnostics."""
    # Import and run the diagnose script
    diagnose_path = os.path.join(os.path.dirname(__file__), "scripts", "diagnose.py")
    if os.path.exists(diagnose_path):
        exec(open(diagnose_path).read())
    else:
        print("❌ diagnose.py not found at:", diagnose_path)


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Agentic RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py ingest ./data/raw
  python app.py ask "What is chunking?"
  python app.py chat
  python app.py agent "Compare fixed and semantic chunking"
  python app.py agent-chat
  python app.py eval
  python app.py diagnose
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("paths", nargs="+", help="File or directory paths")

    # ask (basic RAG)
    ask_parser = subparsers.add_parser("ask", help="Ask a question (basic RAG)")
    ask_parser.add_argument("question", help="Your question")

    # chat (basic RAG)
    subparsers.add_parser("chat", help="Interactive Q&A (basic RAG)")

    # agent (single question)
    agent_parser = subparsers.add_parser("agent", help="Ask using agentic pipeline")
    agent_parser.add_argument("question", help="Your question")

    # agent-chat (interactive with memory)
    subparsers.add_parser("agent-chat", help="Interactive agentic chat with memory")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation metrics")
    eval_parser.add_argument("--dataset", help="Path to eval dataset JSON", default=None)

    # diagnose
    subparsers.add_parser("diagnose", help="Run pipeline diagnostics")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "agent": cmd_agent,
        "agent-chat": cmd_agent_chat,
        "eval": cmd_eval,
        "diagnose": cmd_diagnose,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
