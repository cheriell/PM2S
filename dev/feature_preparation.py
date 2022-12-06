

from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.abspath('./'))
import argparse
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import pretty_midi as pm
import pickle
import numpy as np
import shutil

from data.data_utils import *


class FeaturePreparation():

    def __init__(self, dataset_folder, feature_folder, workers):
        self.dataset_folder = dataset_folder
        self.feature_folder = feature_folder
        self.workers = workers

        # dataset folders
        [self.ASAP, self.A_MAPS, self.CPM, self.ACPAS] = self.dataset_folder
        # feature folder
        Path(self.feature_folder).mkdir(parents=True, exist_ok=True)

    def prepare_metadata(self):
        print('INFO: Preparing metadata...')

        # =========== get ACPAS metadata ===========
        print('INFO: Getting ACPAS metadata')
        ACPAS_metadata_S = pd.read_csv(str(Path(self.ACPAS, 'metadata_S.csv')))
        ACPAS_metadata_R = pd.read_csv(str(Path(self.ACPAS, 'metadata_R.csv')))
        ACPAS_metadata = pd.concat([
            ACPAS_metadata_R[ACPAS_metadata_R['source'] == 'MAPS'], 
            ACPAS_metadata_S
        ], ignore_index=True)

        # ============ create metadata ============
        print('INFO: Creating metadata')
        metadata = pd.DataFrame(columns=[
            'performance_id',
            'piece_id',
            'source',
            'split',
            'midi_perfm',
            'annot_file',
            'feature_file',
            'performance_MIDI_external',
        ])

        format_path = lambda x: x.format(ASAP=self.ASAP, A_MAPS=self.A_MAPS, CPM=self.CPM)
        test_ids = set()

        for _, row in ACPAS_metadata.iterrows():
            # performance_id
            performance_id = row['performance_id']
            # piece_id
            piece_id = row['piece_id']
            # source
            source = row['source']
            # split
            split = '--'
            if performance_id[0] == 'R' and source == 'MAPS':
                subset = row['performance_MIDI_external'][:-4].split('_')[-1]
                if subset == 'ENSTDkCl':
                    split = 'test'
                    test_ids.add(piece_id)
            if split == '--':
                if piece_id in test_ids:
                    pass
                elif piece_id % 10 != 0:
                    split = 'train'
                else:
                    split = 'valid'
            # midi_perfm
            if str(row['performance_MIDI_external']) == 'nan':
                midi_perfm = '--'
                print('WARNING: Performance {} has no MIDI file'.format(row['performance_id']))
            else:
                midi_perfm = format_path(row['performance_MIDI_external'])
            # annot_file
            if row['source'] == 'ASAP':
                if str(row['performance_annotation_external']) == 'nan':
                    annot_file = '--'
                    print('WARNING: Performance {} has no annotation file'.format(row['performance_id']))
                else:
                    annot_file = format_path(row['performance_annotation_external'])
            else:
                annot_file = '--'
            # feature_file
            feature_file = Path(self.feature_folder, '{}.pkl'.format(performance_id))

            metadata = metadata.append({
                'performance_id': performance_id,
                'piece_id': piece_id,
                'source': source,
                'split': split,
                'midi_perfm': midi_perfm,
                'annot_file': annot_file,
                'feature_file': str(feature_file),
                'performance_MIDI_external': row['performance_MIDI_external'],
            }, ignore_index=True)

        # ======== save metadata ==========
        metadata.to_csv('metadata/metadata.csv', index=False)
        print('INFO: Metadata saved to {}'.format(Path(self.feature_folder, 'metadata.csv')))
        
        self.metadata = metadata

    def load_metadata(self):
        print('INFO: Loading metadata...')
        self.metadata = pd.read_csv(str(Path(self.feature_folder, 'metadata.csv')))

    def print_statistics(self):
        print('INFO: Printing dataset statistics')

        # ========== number of distinct pieces ==========
        print('INFO: Get number of distinct pieces')
        pieces_train = set(self.metadata[self.metadata['split'] == 'train']['piece_id'].unique())
        pieces_valid = set(self.metadata[self.metadata['split'] == 'valid']['piece_id'].unique())
        pieces_test = set(self.metadata[self.metadata['split'] == 'test']['piece_id'].unique())

        n_pieces_train = len(pieces_train)
        n_pieces_valid = len(pieces_valid)
        n_pieces_test = len(pieces_test)
        n_pieces = n_pieces_train + n_pieces_valid + n_pieces_test
        
        # =========== number of performances ===========
        print('INFO: Get number of performances')

        def get_performance_label(row):
            if row['split'] == 'test':
                print('-'.join([row['source'], str(row['piece_id'])]))
                
            if row['source'] == 'ASAP':
                return row['performance_MIDI_external']
            else:
                return '-'.join([row['source'], str(row['piece_id'])])

        performance_labels = {'train': set(), 'valid': set(), 'test': set()}
        perfms = {'train': set(), 'valid': set(), 'test': set()}

        for _, row in self.metadata.iterrows():
            if row['split'] == '--':
                continue

            performance_label = get_performance_label(row)
            if performance_label not in performance_labels[row['split']]:
                performance_labels[row['split']].add(performance_label)
                perfms[row['split']].add(row['performance_id'])
            
        perfms_all = perfms['train'] | perfms['valid'] | perfms['test']
        n_perfms_train = len(perfms['train'])
        n_perfms_valid = len(perfms['valid'])
        n_perfms_test = len(perfms['test'])
        n_perfms = n_perfms_train + n_perfms_valid + n_perfms_test

        # ========== duration & number of notes ==========
        print('INFO: Get duration & number of notes')

        def cache_duration_n_notes(row):
            if row['performance_id'] not in perfms_all:
                return
            
            print('INFO: Get duration & number of notes for performance {}'.format(row['performance_id']))

            midi_data = pm.PrettyMIDI(row['midi_perfm'])
            duration = midi_data.get_end_time()
            n_notes = np.sum([len(midi_data.instruments[i].notes) for i in range(len(midi_data.instruments))])

            cache_file = Path(self.feature_folder, 'temp', row['performance_id']+'.pkl')
            pickle.dump((duration, n_notes), open(str(cache_file), 'wb'))

        Path(self.feature_folder, 'temp').mkdir(parents=True, exist_ok=True)
        rows = [row for _, row in self.metadata.iterrows()]
        pool = Pool(processes=self.workers)
        pool.map(cache_duration_n_notes, rows)

        duration_train, duration_valid, duration_test = 0, 0, 0
        n_notes_train, n_notes_valid, n_notes_test = 0, 0, 0

        for _, row in self.metadata.iterrows():
            if row['performance_id'] not in perfms_all:
                continue

            cache_file = Path(self.feature_folder, 'temp', row['performance_id']+'.pkl')
            duration, n_notes = pickle.load(open(str(cache_file), 'rb'))

            if row['performance_id'] in perfms['train']:
                duration_train += duration
                n_notes_train += n_notes
            elif row['performance_id'] in perfms['valid']:
                duration_valid += duration
                n_notes_valid += n_notes
            elif row['performance_id'] in perfms['test']:
                duration_test += duration
                n_notes_test += n_notes

        shutil.rmtree(str(Path(self.feature_folder, 'temp')))

        duration_all = duration_train + duration_valid + duration_test
        n_notes_all = n_notes_train + n_notes_valid + n_notes_test

        # ======== print dataset statistics ==========
        print('\n\t=================== Dataset Statistics ====================')
        print('\t\t\tTrain\t\tValid\t\tTest\t\tAll')
        print('\tn_pieces:\t{}\t\t{}\t\t{}\t\t{}'.format(n_pieces_train, n_pieces_valid, \
                            n_pieces_test, n_pieces))
        print('\tn_perfms:\t{}\t\t{}\t\t{}\t\t{}'.format(n_perfms_train, n_perfms_valid, \
                            n_perfms_test, n_perfms))
        print('\tduration (h):\t{:.1f}\t\t{:.1f}\t\t{:.1f}\t\t{:.1f}'.format(duration_train/3600, \
                            duration_valid/3600, duration_test/3600, duration_all/3600))
        print('\tn_notes (k):\t{:.1f}\t\t{:.1f}\t\t{:.1f}\t\t{:.1f}\n'.format(n_notes_train/1000, \
                            n_notes_valid/1000, n_notes_test/1000, n_notes_all/1000))
        
    def prepare_features(self):
        print('INFO: Preparing features...')

        def prepare_one_feature(row):
            print('INFO: Preparing feature {}'.format(row['performance_id']))

            if row['source'] == 'ASAP':
                # get note sequence
                note_sequence = get_note_sequence_from_midi(row['midi_perfm'])
                # get annotations dict (beats, downbeats, key signatures, time signatures)
                annotations = get_annotations_from_annot_file(row['annot_file'])
            else:
                # get note sequence and annotations dict
                # (beats, downbeats, key signatures, time signatures, musical onset times, note value in beats, hand parts)
                note_sequence, annotations = get_note_sequence_and_annotations_from_midi(row['midi_perfm'])
            
            pickle.dump((note_sequence, annotations), open(row['feature_file'], 'wb'))

        if self.workers > 0:
            # prepare features with multiprocessing
            rows = [row for _, row in self.metadata.iterrows()]
            pool = Pool(self.workers)
            pool.map(prepare_one_feature, rows)
        else:
            # prepare features without multiprocessing
            for _, row in self.metadata.iterrows():
                prepare_one_feature(row)

        print('INFO: Features prepared')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders in the order \
                        of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--feature_folder', type=str, help='Path to the feature folder')
    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing, 0 for not using \
                        multiprocessing, minus for using default number of workers', default=mp.cpu_count())
    args = parser.parse_args()

    # ========= input check =========
    # dataset_folder
    if args.dataset_folder is None or len(args.dataset_folder) != 4:
        raise ValueError('dataset_folder should be a list of 4 string paths')
    else:
        for folder in args.dataset_folder:
            if not Path(folder).exists():
                raise ValueError('dataset_folder {} does not exist'.format(folder))
    # workers
    if args.workers < 0:
        args.workers = mp.cpu_count()
        print('INFO: Using default number of workers: {}'.format(args.workers))
    elif args.workers > mp.cpu_count():
        args.workers = mp.cpu_count()
        print('WARNING: Number of workers is greater than the number of CPU cores, setting to {}'.format(args.workers))
    else:
        pass

    # ========= feature preparation =========
    featprep = FeaturePreparation(args.dataset_folder, args.feature_folder, args.workers)

    featprep.prepare_metadata()
    featprep.print_statistics()
    featprep.prepare_features()