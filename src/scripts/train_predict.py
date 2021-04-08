import argparse
from common.loadData import load_all_data
from model.roberta_model import train_predict_model, predict


feature_stance = ['polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']
feature_related = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs']
feature_all = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs',
                       'polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']


def main(parser):
    args = parser.parse_args()

    type_class = args.type_class
    use_cuda = args.use_cuda
    not_use_feature = args.not_use_feature
    training_set = args.training_set
    test_set = args.test_set
    model_dir = args.model_dir
    feature = []
    if not not_use_feature:
        if type_class == 'stance':
            feature = feature_stance
        elif type_class == 'related':
            feature = feature_related
        elif type_class == 'all':
            feature = feature_all

    if model_dir == "":
        df_test = load_all_data(test_set, type_class, feature)
        df_train = load_all_data(training_set, type_class, feature)
        train_predict_model(df_train, df_test, True, use_cuda, len(feature))
    else:
        df_test = load_all_data(test_set, type_class, feature)
        predict(df_test, use_cuda, model_dir, len(feature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--type_class", choices=['related', 'stance', 'all'],
                        default='related', help="This parameter is used to choose the type of "
                                                "clasificator (related, stance, all).")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be used if cuda is present.")

    parser.add_argument("--not_use_feature",
                        default=False,
                        action='store_true',
                        help="This parameter should be used if you don't want to train with the external features.")

    parser.add_argument("--training_set",
                        default="/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="This parameter is the relatis_type_contradictionive dir of training set.")

    parser.add_argument("--test_set",
                        default="/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    main(parser)
