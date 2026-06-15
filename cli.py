import json
import sys
from query_engine import ask_question


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python app.py \"<question>\"")
        sys.exit(1)

    try:
        response = ask_question(question)
        print(json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()