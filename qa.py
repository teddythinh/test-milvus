import argparse
from datasets import Dataset
from transformers import pipeline
from pymilvus import connections, utility, Collection
from build_database import tokenize_question, quest_embedding, search
from build_database import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    REPLICA_NUMBER,
    TOKENIZATION_BATCH_SIZE, 
    INFERENCE_BATCH_SIZE
)

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
if utility.has_collection(COLLECTION_NAME):
    qa_collection = Collection(COLLECTION_NAME)
    qa_collection.load(replica_number=REPLICA_NUMBER)
else:
    raise RuntimeError

PIPELINE_NAME = 'question-answering'
MODEL_NAME = 'thangduong0509/distilbert-finetuned-squadv2'
qa_pipeline = pipeline(PIPELINE_NAME, model=MODEL_NAME)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    questions = {'question': [f'{args.question}']}
    question_dataset = Dataset.from_dict(questions)

    question_dataset = question_dataset.map(
        tokenize_question, 
        batched=True, 
        batch_size=TOKENIZATION_BATCH_SIZE
    )
    question_dataset.set_format(
        'torch', 
        columns=['input_ids', 'attention_mask'], 
        output_all_columns=True
    )
    question_dataset = question_dataset.map(
        quest_embedding, 
        remove_columns=['input_ids', 'attention_mask'], 
        batched=True, 
        batch_size=INFERENCE_BATCH_SIZE
    )

    retrieval_results = question_dataset.map(search, batched=True, batch_size=1)
    for result in retrieval_results:
        print()
        print('Input Question:')
        print(result['question'])
        print()
        for rank_idx, candidate in enumerate(zip(result['similar_question'], result['context'], result['distance'])):
            context = candidate[1]
            distance = candidate[2].tolist()
            predicted_answer = qa_pipeline(
                context=context,
                question=args.question
            )

            print(f'Relevant Context Rank {rank_idx+1}:')
            print(f'Context: {context}')
            print(f'Score: {distance}')
            print(f'Predicted Answer: {predicted_answer}')
            print()

if __name__ == '__main__':
    main()
    qa_collection.release()