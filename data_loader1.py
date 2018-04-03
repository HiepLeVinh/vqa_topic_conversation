from os.path import isfile, join
import re
import numpy as np
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


def prepare_training_data(num_topic):

    with open("data/dataset_remove.pkl", "rb") as f:
        data = pickle.load(f)

    conversations = [" ".join((v["utterance"], v["response"])) for k, v in data.items()]
    topics = [extract_object(v["obj"])[-1] for k, v in data.items()]

    topic_vocab = make_topic_vocab(topics, num_topic)
    print("topic_vocab", topic_vocab)

    conversation_vocab, max_conversation_length = make_conversation_vocab(conversations, topics, topic_vocab)
    print("conversation_vocab", conversation_vocab)
    print("max_conversation_length", max_conversation_length)

    print("Extracting training data")
    word_regex = re.compile(r'\w+')
    training_data = []
    for i, image_id in enumerate(data.keys()):
        topic = topics[i]
        if topic in topic_vocab:
            conversation = conversations[i]
            conversation_words = re.findall(word_regex, conversation)
            # conversation: fil id of the question word (not the word), same length with max conversation length
            # keep zero at the emd if length less than max
            conversation_ids = np.zeros(max_conversation_length, dtype=int)
            for index, word in enumerate(conversation_words):
                conversation_ids[index] = conversation_vocab[word]

            training_data.append({
                'image_id': image_id,
                'topic': topic_vocab[topic],  # id, not the word
                'conversation': conversation_ids
            })

    print("training_data", training_data)
    print("training data len", len(training_data))

    total_size = len(training_data)
    training_data_size = int(0.7 * total_size)

    data = {
        "training": training_data[:training_data_size],
        "validation": training_data[training_data_size:],
        "topic_vocab": topic_vocab,
        "conversation_vocab": conversation_vocab,
        "max_conversation_length": max_conversation_length
    }
    print("Saving data")
    with open("data/data_file", 'wb') as f:
        pickle.dump(data, f)

    with open("data/data_file", 'rb') as f:
        data = pickle.load(f)
        print("topic_vocab", data["topic_vocab"])

    return data


def load_data(data_dir='data'):
    qa_data_file = join(data_dir, 'data_file')
    print(qa_data_file)

    if isfile(qa_data_file):
        with open(qa_data_file, 'rb') as f:
            data = pickle.load(f)
            return data


def get_question_answer_vocab(version=2, data_dir='Data'):
    vocab_file = join(data_dir, 'vocab_file{}.pkl'.format(version))
    vocab_data = pickle.load(open(vocab_file))
    return vocab_data


def make_topic_vocab(topics, num_topic):
    topic_frequency = Counter()
    for topic in topics:
        topic_frequency[topic] += 1

    topic_frequency_tuples = [(-frequency, topic) for topic, frequency in topic_frequency.items()]
    topic_frequency_tuples.sort()
    topic_frequency_tuples_top = topic_frequency_tuples[0:num_topic - 1]
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

    topic_vocab['UNK'] = num_topic - 1
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
                # Check word duplicate to topic, if found, stop everything
                if word in topic:
                    print("Duplicate caution word in topic!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(topic, word, conversation)
                    return [], []
                conversation_word_frequency[word] += 1
            if len(conversation_words) > max_conversation_length:
                max_conversation_length = len(conversation_words)

    print("conversation_word_frequency", conversation_word_frequency)
    word_freq_threshold = 0
    conversation_word_vocab = {}
    index = 0
    for word, frequency in conversation_word_frequency.items():
        if frequency > word_freq_threshold:
            # +1 for accounting the zero padding for batch training, index 0 is used to fill missing length conversation
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
