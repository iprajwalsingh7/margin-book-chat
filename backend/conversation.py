"""Bounded conversation context and follow-up retrieval."""
import re
from backend import retrieval


def completed_context(turns):
    return [{'question': turn['question'], 'answer': turn['result']['answer'][:4000]}
            for turn in turns if turn['status'] == 'complete' and turn.get('result', {}).get('supported')][-3:]


def is_followup(question):
    return bool(re.search(r'\b(?:it|that|those|these|they|them|this idea|more|simpler|simple terms|example)\b', question, re.I)
                or re.match(r'^(?:why|how so|explain again)[?! .]*$', question, re.I))


def retrieve_for_turn(book, question, turns):
    successful = [turn for turn in turns if turn['status'] == 'complete' and turn.get('result', {}).get('supported')]
    if not successful or not is_followup(question):
        return retrieval.retrieve(book, question)
    previous = successful[-1]
    query = f"{question}\nPrevious question: {previous['question']}\nPrevious answer: {previous['result']['answer'][:1000]}"
    chunks = retrieval.retrieve(book, query)
    known = {chunk['id']: chunk for chunk in book['chunks']}
    ids = {chunk['id'] for chunk in chunks}
    for source in previous['result']['sources'][:3]:
        if source['id'] in known and source['id'] not in ids:
            chunks.append(known[source['id']])
            ids.add(source['id'])
    return chunks[:18 if retrieval.is_overview_question(query) else 8]
