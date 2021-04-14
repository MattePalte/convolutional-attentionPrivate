#!/home/paltenmo/projects/AllamanisCodeSummarization/convolutional-attention/condaenv/bin/python2.7

from convolutional_attention.copy_conv_rec_learner import ConvolutionalCopyAttentionalRecurrentLearner
import numpy as np
import json
import sys
import os

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    print "Usage <model.pkl> <dataset.json> <output.json>"
    #    sys.exit(-1)

    projects = [
        'elasticsearch',
        'cassandra',
        'gradle',
        'hadoop-common',
        'hibernate-orm',
        'intellij-community',
        'liferay-portal',
        'presto',
        'spring-framework',
        'wildfly'
    ]

    ROOT = '../dataset_convolutional-attention/json/'
    all_datasets = os.listdir(ROOT)
    for p in projects:
        test = [f for f in all_datasets if 'test' in f and p in f][0]
        train = [f for f in all_datasets if 'train' in f and p in f][0]
        #params["test_file"] = os.path.join(ROOT, test)
        #params["train_file"] = os.path.join(ROOT, train)
        input_file = os.path.join(ROOT, test)
        sys.stdout = open('visualizer_' + p + '.log', 'w')
        print input_file

        learner = ConvolutionalCopyAttentionalRecurrentLearner.load(p + "_copy_convolutional_att_rec_model.pkl")
        dataset = input_file
        output_file = "GPU_" + p + "_attentions.json"

        data, original_names = learner.naming_data.get_data_in_recurrent_copy_convolution_format(dataset, learner.padding_size)
        name_targets, code_sentences, code, target_is_unk, copy_vectors = data

        all_data = []
        for i in xrange(len(name_targets)):
            for j in xrange(1, len(name_targets[i])):
                suggestion_data = {}

                current_prefix = name_targets[i][:j]
                suggestion_data["original_name"] = original_names[i]

                if j == len(name_targets[i]) -1:  # END token
                    target_subtoken = learner.naming_data.all_tokens_dictionary.get_name_for_id(name_targets[i][j])
                else:
                    target_subtoken = original_names[i].split(",")[j-1]
                suggestion_data["target subtoken"] = target_subtoken

                copy_weights, copy_prob, subtoken_probs = learner.model.copy_probs(current_prefix, code_sentences[i])

                copy_prob, suggestions, subtoken_probs = learner.get_suggestions_for_next_subtoken(code[i], code_sentences[i], current_prefix)
                suggestion_data["suggestions"] = {k: np.exp(subtoken_probs[k]) for k in suggestions[:20]}

                suggestion_data["att_vector"] = [p for p in learner.model.attention_weights(current_prefix, code_sentences[i])[-1, 0]]
                suggestion_data["copy_vector"] = [p for p in copy_weights[-1]]
                suggestion_data["copy_prob"] = float(copy_prob)
                suggestion_data["tokens"] = code[i]
                suggestion_data["is_unk"] = [learner.naming_data.all_tokens_dictionary.is_unk(t) for t in code[i]]
                all_data.append(suggestion_data)


        with open(output_file, 'w') as f:
            json.dump(all_data, f, cls=MyEncoder)
