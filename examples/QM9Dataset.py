import collections
import contextlib
import json
import os
import random
import subprocess
import tarfile

import flowws
from flowws import Argument as Arg
import gtar
import numpy as np

ATTR_NAMES = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo',
              'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

@flowws.add_stage_arguments
class QM9Dataset(flowws.Stage):
    """Load the QM9 dataset.

    """

    ARGS = [
        Arg('cache_dir', '-c', str, '/tmp',
            help='Directory to store dataset'),
        Arg('y_attribute', '-y', str, 'U0',
            help='Attribute to expose for prediction'),
        Arg('n_train', '-n', int, 100000,
            help='Number of examples to use for training set'),
        Arg('n_val', None, int, 1000,
            help='Number of examples to use for validation set'),
        Arg('seed', '-s', int, 13,
            help='Random number seed to use'),
    ]

    def run(self, scope, storage):
        random.seed(self.arguments['seed'])

        gtar_fname = self._get_gtar_filename()

        max_atoms = 0
        type_map = collections.defaultdict(lambda: len(type_map))
        type_map['NULL']

        dataset = {}
        with gtar.GTAR(gtar_fname, 'r') as traj:
            for (frame, (type_names, types, positions, attrs)) in traj.recordsNamed(
                    ['type_names.json', 'type', 'position', 'attributes.json']):
                type_names = json.loads(type_names)
                attrs = json.loads(attrs)

                new_types = [type_map[type_names[t]] for t in types]
                dataset[frame] = (positions, new_types,
                                  attrs[self.arguments['y_attribute']])
                max_atoms = max(max_atoms, len(types))

        assert len(dataset)
        N_train = self.arguments['n_train']
        N_val = self.arguments['n_val']

        frames = list(sorted(dataset))
        random.shuffle(frames)
        train_frames, frames = frames[:N_train], frames[N_train:]
        val_frames, test_frames = frames[:N_val], frames[N_val:]
        N_test = len(test_frames)

        num_types = len(type_map)

        datasets = {}
        for name in ['train', 'val', 'test']:
            dset_xs, dset_ts, dset_ys = [], [], []
            frames = locals()['{}_frames'.format(name)]
            for frame in frames:
                (xs, ts, ys) = self.get_encoding(dataset[frame], max_atoms, type_map)
                dset_xs.append(xs)
                dset_ts.append(ts)
                dset_ys.append(ys)

            dset_xs = np.array(dset_xs)
            dset_ts = np.array(dset_ts)
            dset_ys = np.array(dset_ys)

            datasets[name] = (dset_xs, dset_ts, dset_ys)

        yref = datasets['train'][-1]
        mu = np.mean(yref)
        sigma = np.std(yref)

        for (key, vals) in list(datasets.items()):
            vals = list(vals)
            vals[-1] = (vals[-1] - mu)/sigma
            datasets[key] = vals

        scope['y_scale'] = sigma
        scope['neighborhood_size'] = max_atoms
        scope['num_types'] = num_types
        scope['x_train'] = datasets['train'][:2]
        scope['y_train'] = datasets['train'][-1]
        scope['x_test'] = datasets['test'][:2]
        scope['y_test'] = datasets['test'][-1]
        scope['validation_data'] = (datasets['val'][:2], datasets['val'][-1])
        scope['type_map'] = type_map
        scope.setdefault('metrics', []).extend(['mae'])

    def _convert_to_gtar(self, tar_name, gtar_name):
        with contextlib.ExitStack() as stack:
            tf = stack.enter_context(tarfile.open(tar_name, 'r'))
            traj = stack.enter_context(gtar.GTAR(gtar_name, 'w'))

            file_count = 0
            for entry in tf:
                if entry.isfile():
                    self._save_entry(file_count, tf.extractfile(entry), traj)
                    file_count += 1

    def _get_gtar_filename(self):
        url = 'https://figshare.com/ndownloader/files/3195389'
        source_fname = 'dsgdb9nsd.xyz.tar.bz2'
        dest_fname = os.path.join(self.arguments['cache_dir'], 'qm9_data.zip')

        if not os.path.exists(dest_fname):
            output_name = os.path.join(self.arguments['cache_dir'], source_fname)
            command = ['wget', '-c', '-O', output_name, url]
            subprocess.check_call(command)
            self._convert_to_gtar(output_name, dest_fname)
            os.remove(output_name)

        return dest_fname

    def _save_entry(self, index, xyz_file, traj):
        text = xyz_file.read().decode().replace('*^', 'e')
        lines = text.splitlines()

        atom_count = int(lines[0])
        attr_line = lines[1]
        attrs = dict(zip(ATTR_NAMES, attr_line.split()))
        coord_lines = lines[2:2 + atom_count]

        atom_type_names = []
        atom_coords = []
        for line in coord_lines:
            line = line.split()
            atom_type_names.append(line[0])
            atom_coords.append(tuple(map(float, line[1:4])))

        all_types = list(sorted(set(atom_type_names)))
        type_map = {t: i for (i, t) in enumerate(all_types)}
        types = [type_map[t] for t in atom_type_names]

        prefix = 'frames/{}/'.format(index)
        traj.writePath(prefix + 'attributes.json', json.dumps(attrs))
        traj.writePath(prefix + 'type_names.json', json.dumps(all_types))
        traj.writePath(prefix + 'type.u32.ind', types)
        traj.writePath(prefix + 'position.f32.ind', atom_coords)

    @staticmethod
    def get_encoding(data, max_atoms, type_map):
        (rs, ts, ys) = data
        types = np.zeros(max_atoms, dtype=np.uint32)
        types[:len(ts)] = ts
        types_onehot = np.eye((len(type_map)))[types]

        positions = np.zeros((max_atoms, 3))
        positions[:len(rs)] = rs

        return positions, types_onehot, float(ys)
