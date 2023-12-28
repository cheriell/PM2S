import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch


def evaluate_time_signature_prediction():

    from data.dataset_time_signature import TimeSignatureDataset
    from pm2s.models.time_signature import CNNTimeSignatureModel
    from modules.utils import f_measure_framewise

    testset = TimeSignatureDataset(
        workspace='/import/c4dm-05/ll307/workspace/PM2S-timesigcorrection', 
        split='test',
    )
    model = CNNTimeSignatureModel()
    model.load_state_dict(torch.load('nohup.out.ts_cnn.pth'))
    model = model.to('cuda')
    model.eval()

    targets = []
    predictions = []

    for i in range(len(testset)):
        print('testing sample {}/{}'.format(i+1, len(testset)))

        # Get data
        note_sequence, time_sigs, length = testset[i]
        note_sequence = torch.Tensor(note_sequence).to('cuda').float().unsqueeze(0)

        # Get prediction
        ts_probs = model(note_sequence)
        ts_probs = ts_probs.squeeze(0).to('cpu').detach().numpy()  # (seq_len,)
        ts_pred = (ts_probs[:length] > 0.45).astype(int)
        ts_pred = np.bincount(ts_pred).argmax()

        # Get ground truth
        ts_targ = np.bincount(time_sigs[:length].astype(int)).argmax()

        # Append
        predictions.append(ts_pred)
        targets.append(ts_targ)

    # Compute f-measure
    predictions = np.array(predictions)
    targets = np.array(targets)
    print('predictions:\n', predictions)
    print('targets:\n', targets)
    acc, p, r, f = f_measure_framewise(torch.Tensor(targets), torch.Tensor(predictions))
    print('acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}'.format(acc, p, r, f))


if __name__ == '__main__':

    evaluate_time_signature_prediction()
