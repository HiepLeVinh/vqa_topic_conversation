import tensorflow as tf
import vis_model
import data_loader1
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                        help='num_lstm_layers')
    parser.add_argument('--fc7_feature_length', type=int, default=4096,
                        help='fc7_feature_length')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                        help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5,
                        help='image_dropout')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Expochs')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug')
    parser.add_argument('--model_path', type=str, default='data/models/model29.ckpt',
                        help='Model Path')
    parser.add_argument('--version', type=int, default=2,
                        help='VQA data version')

    args = parser.parse_args()
    print("Reading QA DATA")
    # qa_data = data_loader.load_questions_answers(args)
    qa_data = data_loader1.load_data(args.data_dir)

    print("Reading fc7 features")
    fc7_features, image_id_list = data_loader1.load_fc7_features(args.data_dir, 'val')
    print("FC7 features", fc7_features.shape)
    print("image_id_list", image_id_list.shape)

    image_id_map = {}
    for i in range(len(image_id_list)):
        image_id_map[image_id_list[i]] = i

    ans_map = {qa_data['topic_vocab'][ans]: ans for ans in qa_data['topic_vocab']}

    model_options = {
        'num_lstm_layers': args.num_lstm_layers,
        'rnn_size': args.rnn_size,
        'embedding_size': args.embedding_size,
        'word_emb_dropout': args.word_emb_dropout,
        'image_dropout': args.image_dropout,
        'fc7_feature_length': args.fc7_feature_length,
        'lstm_steps': qa_data['max_conversation_length'] + 1,
        'q_vocab_size': len(qa_data['conversation_vocab']),
        'ans_vocab_size': len(qa_data['topic_vocab'])
    }

    model = vis_model.Vis_lstm_model(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    avg_accuracy = 0.0
    total = 0
    saver.restore(sess, args.model_path)

    batch_no = 0
    while (batch_no * args.batch_size) < len(qa_data['validation']):
        sentence, answer, fc7 = get_batch(batch_no, args.batch_size,
                                          fc7_features, image_id_map, qa_data, 'val')

        pred, ans_prob = sess.run([t_prediction, t_ans_probab], feed_dict={
            input_tensors['fc7']: fc7,
            input_tensors['sentence']: sentence,
        })

        batch_no += 1
        if args.debug:
            for idx, p in enumerate(pred):
                print(ans_map[p], ans_map[np.argmax(answer[idx])])

        correct_predictions = np.equal(pred, np.argmax(answer, 1))
        correct_predictions = correct_predictions.astype('float32')
        accuracy = correct_predictions.mean()
        print("Acc", accuracy)
        avg_accuracy += accuracy
        total += 1

    print("Acc", avg_accuracy / total)


def get_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
    qa = None
    if split == 'train':
        qa = qa_data['training']
    else:
        qa = qa_data['validation']

    si = (batch_no * batch_size) % len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    sentence = np.ndarray((n, qa_data['max_conversation_length']), dtype='int32')
    answer = np.zeros((n, len(qa_data['topic_vocab'])))
    fc7 = np.ndarray((n, 4096))

    count = 0

    for i in range(si, ei):
        sentence[count, :] = qa[i]['conversation'][:]
        answer[count, qa[i]['topic']] = 1.0
        fc7_index = image_id_map[qa[i]['image_id']]
        fc7[count, :] = fc7_features[fc7_index][:]
        count += 1

    return sentence, answer, fc7


if __name__ == '__main__':
    main()
