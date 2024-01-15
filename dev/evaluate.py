import warnings
warnings.filterwarnings('ignore')
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--model_state_dict_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()

    if args.feature == 'beat':
        print('Evaluating beat prediction ...')
        from evaluation.beat import evaluate_beat_prediction
        evaluate_beat_prediction(args)

    elif args.feature == 'time_signature':
        print('Evaluating time signature prediction ...')
        from evaluation.time_signature import evaluate_time_signature_prediction
        evaluate_time_signature_prediction(args)
    
    elif args.feature == 'key_signature':
        print('Evaluating key signature prediction ...')
        from evaluation.key_signature import evaluate_key_signature_prediction
        evaluate_key_signature_prediction(args)

    elif args.feature == 'hand_part':
        print('Evaluating hand part prediction ...')
        from evaluation.hand_part import evaluate_hand_part_prediction
        evaluate_hand_part_prediction(args)

    elif args.feature == 'quantisation':
        print('Evaluating quantisation prediction ...')
        from evaluation.quantisation import evaluate_quantisation_prediction
        evaluate_quantisation_prediction(args)

    else:
        raise Exception('Incorrect argument!')