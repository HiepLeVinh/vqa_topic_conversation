import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
from utils import extract_object
from collections import Counter

"""
Data format from dataset.pkl
{
"image_id1": {"utterance": utterance, "response": response, "obj": obj,
                "utterance_person_info": utterance_person_info, "response_person_info": response_person_info},
"image_id2": ...
}
image_id: long
utterance, response - text
obj (topic), utterance_person_info, response_person_info -  x; y; w; h; id; name
"""


def prepare_training_data():

    with open("data/dataset.pkl", "rb") as f:
        data = pickle.load(f)

    conversations = [" ".join((v["utterance"], v["response"])) for k, v in data.items()]
    topics = [extract_object(v["obj"])[-1] for k, v in data.items()]

    topic_vocab = make_topic_vocab(topics)
    print("topic_vocab", topic_vocab)

    conversation_vocab, max_conversation_length = make_conversation_vocab(conversations, topics, topic_vocab)
    # print("conversation_vocab", conversation_vocab)
    # print("max_conversation_length", max_conversation_length)

    # answer_vocab = make_answer_vocab(answers)
    # question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
    # print "Max Question Length", max_question_length
    # word_regex = re.compile(r'\w+')
    # training_data = []
    # for i, question in enumerate(t_questions['questions']):
    #     ans = t_answers['annotations'][i]['multiple_choice_answer']
    #     if ans in answer_vocab:
    #         training_data.append({
    #             'image_id': t_answers['annotations'][i]['image_id'],
    #             'question': np.zeros(max_question_length),
    #             'answer': answer_vocab[ans] # id, not the word
    #         })
    #         question_words = re.findall(word_regex, question['question'])
    #
    #         # question: fill id of the question word (not the word), same length of max question length,
    #         # fill zero if length less than max
    #         base = max_question_length - len(question_words)
    #         for i in range(0, len(question_words)):
    #             training_data[-1]['question'][base + i] = question_vocab[question_words[i]]
    #
    # print "Training Data", len(training_data)
    # val_data = []
    # for i, question in enumerate(v_questions['questions']):
    #     ans = v_answers['annotations'][i]['multiple_choice_answer']
    #     if ans in answer_vocab:
    #         val_data.append({
    #             'image_id': v_answers['annotations'][i]['image_id'],
    #             'question': np.zeros(max_question_length),
    #             'answer': answer_vocab[ans]
    #         })
    #         question_words = re.findall(word_regex, question['question'])
    #
    #         base = max_question_length - len(question_words)
    #         for i in range(0, len(question_words)):
    #             val_data[-1]['question'][base + i] = question_vocab[question_words[i]]
    #
    # print "Validation Data", len(val_data)
    #
    # data = {
    #     'training': training_data,
    #     'validation': val_data,
    #     'answer_vocab': answer_vocab,
    #     'question_vocab': question_vocab,
    #     'max_question_length': max_question_length
    # }
    #
    # print "Saving qa_data"
    # with open(qa_data_file, 'wb') as f:
    #     pickle.dump(data, f)
    #
    # with open(vocab_file, 'wb') as f:
    #     vocab_data = {
    #         'answer_vocab': data['answer_vocab'],
    #         'question_vocab': data['question_vocab'],
    #         'max_question_length': data['max_question_length']
    #     }
    #     pickle.dump(vocab_data, f)

    return data


def load_questions_answers(version=2, data_dir='Data'):
    qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
    print(qa_data_file)

    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = pickle.load(f)
            return data


def get_question_answer_vocab(version=2, data_dir='Data'):
    vocab_file = join(data_dir, 'vocab_file{}.pkl'.format(version))
    vocab_data = pickle.load(open(vocab_file))
    return vocab_data


def make_topic_vocab(topics):
    top_n = 50
    topic_frequency = Counter()
    for topic in topics:
        topic_frequency[topic] += 1

    topic_frequency_tuples = [(-frequency, topic) for topic, frequency in topic_frequency.items()]
    topic_frequency_tuples.sort()
    topic_frequency_tuples_top = topic_frequency_tuples[0:top_n - 1]
    print("total topic", len(topics))
    print("topic_frequency_sort", topic_frequency_tuples_top)
    total_popular_topic_count = sum([-x[0] for x in topic_frequency_tuples_top])
    print("total_popular_topic_count", total_popular_topic_count)
    print("total_UNK", len(topics) - total_popular_topic_count)
    print("total One", len([x[0] for x in topic_frequency_tuples if x[0] == -1]))

    topic_vocab = {}
    for i, topic_freq in enumerate(topic_frequency_tuples_top):
        topic = topic_freq[1]
        topic_vocab[topic] = i

    topic_vocab['UNK'] = top_n - 1
    return topic_vocab


def make_conversation_vocab(conversations, topics, topic_vocab):
    word_regex = re.compile(r'\w+')
    conversation_word_frequency = Counter()

    max_conversation_length = 0
    for i, conversation in enumerate(conversations):
        topic = topics[i]
        # Just keep the popular topics
        if topic in topic_vocab:
            conversation_words = re.findall(word_regex, conversation)
            for word in conversation_words:
                conversation_word_frequency[word] += 1
            if len(conversation_words) > max_conversation_length:
                max_conversation_length = len(conversation_words)

    word_freq_threshold = 0
    conversation_word_vocab = {}
    index = 0
    for word, frequency in conversation_word_frequency.items():
        if frequency > word_freq_threshold:
            # +1 for accounting the zero padding for batch training
            # Set index for conversation word
            conversation_word_vocab[word] = index + 1
            index += 1

    conversation_word_vocab['UNK'] = len(conversation_word_vocab) + 1

    return conversation_word_vocab, max_conversation_length


def load_fc7_features(data_dir, split):
    import h5py
    fc7_features = None
    image_id_list = None
    with h5py.File(join(data_dir, (split + '_fc7.h5')), 'r') as hf:
        fc7_features = np.array(hf.get('fc7_features'))
    with h5py.File(join(data_dir, (split + '_image_id_list.h5')), 'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return fc7_features, image_id_list


prepare_training_data()
