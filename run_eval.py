import json
import time
from pathlib import Path

from query_engine import ask_question

EVAL_FILE = Path("eval/questions.json")


def load_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalized_category(category: str) -> str:
    return category.strip().upper()


def evaluate():
    questions = load_questions()
    results = []
    totals = {"ANSWERABLE": 0, "UNANSWERABLE": 0, "FALSE_PREMISE": 0, "ADVERSARIAL": 0}
    correct = 0
    refusal_count = 0
    true_refusal = 0
    false_premise_correct = 0
    start = time.time()

    for item in questions:
        question = item["question"]
        expected = normalized_category(item["expected"])
        response = ask_question(question)
        elapsed = response.get("latency", 0)
        if elapsed == 0:
            elapsed = 0.0
        results.append({"question": question, "expected": expected, "response": response})
        totals[expected] += 1

        predicted = response["type"]
        if expected == "ANSWERABLE" and predicted == "ANSWER":
            correct += 1
        if expected == "UNANSWERABLE" and predicted == "REFUSAL":
            correct += 1
        if expected == "FALSE_PREMISE" and predicted == "FALSE_PREMISE":
            correct += 1
            false_premise_correct += 1
        if expected == "ADVERSARIAL" and predicted == "REFUSAL":
            correct += 1

        if predicted == "REFUSAL":
            refusal_count += 1
            if expected in {"UNANSWERABLE", "FALSE_PREMISE", "ADVERSARIAL"}:
                true_refusal += 1

    total_time = time.time() - start
    average_latency = total_time / len(questions)
    refusal_precision = true_refusal / refusal_count if refusal_count else 0.0
    refusal_recall = true_refusal / (totals["UNANSWERABLE"] + totals["FALSE_PREMISE"] + totals["ADVERSARIAL"]) if totals["UNANSWERABLE"] + totals["FALSE_PREMISE"] + totals["ADVERSARIAL"] else 0.0
    false_premise_accuracy = false_premise_correct / totals["FALSE_PREMISE"] if totals["FALSE_PREMISE"] else 0.0

    print("EVALUATION RESULTS")
    print("------------------")
    print(f"Total questions: {len(questions)}")
    print(f"Accuracy: {correct}/{len(questions)} = {correct/len(questions):.2f}")
    print(f"Refusal precision: {refusal_precision:.2f}")
    print(f"Refusal recall: {refusal_recall:.2f}")
    print(f"False premise accuracy: {false_premise_accuracy:.2f}")
    print(f"Average latency per query: {average_latency:.2f} seconds")
    print(f"Cost estimate: 0.00 (local open-source embedding + reranking stack)")

    return results


if __name__ == "__main__":
    evaluate()

# getting very poor results on first go...its a dry run tho
# Total questions: 48 Accuracy: 16/48 = 0.33 Refusal precision: 0.48 Refusal recall: 1.00
# False premise accuracy: 0.00 Average latency: 0.34s