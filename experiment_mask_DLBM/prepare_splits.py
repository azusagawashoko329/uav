#!/usr/bin/env python3
"""Create and audit fixed 1/2 + 1/4 + 1/4 splits inside every subvideo.

Each subvideo is independently and deterministically shuffled.  Its shuffled frames are divided
into four near-equal folds; folds 0+1 form ``probe_train``, fold 2 forms ``boundary_train``, and
fold 3 forms ``boundary_decision``.  The three manifests are disjoint and cover every train frame.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {
    'bike': 'Bike Roundabout',
    'crossroads': 'Crossroads',
    'farmland': 'Farmland Inspection',
    'highway': 'Highway',
    'railway': 'Railway Inspection',
    'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
SPLITS = ('probe_train', 'boundary_train', 'boundary_decision')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def stable_subvideo_seed(seed, scene, subvideo):
    token = f'{seed}\0{scene}\0{subvideo}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:8], byteorder='big')


def quarter_sizes(length):
    quotient, remainder = divmod(length, 4)
    # Extra frames are assigned in ratio-preserving order.  In particular, for n=4q+2 we use
    # [q+1, q, q+1, q], so probe (folds 0+1) receives exactly 2q+1 rather than 2q+2 frames.
    extra_order = (0, 2, 3, 1)
    sizes = [quotient] * 4
    for index in extra_order[:remainder]:
        sizes[index] += 1
    return sizes


def split_subvideo(relative_frames, seed):
    shuffled = list(relative_frames)
    random.Random(seed).shuffle(shuffled)
    sizes = quarter_sizes(len(shuffled))
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    folds = [shuffled[offsets[i]:offsets[i + 1]] for i in range(4)]
    return {
        'probe_train': sorted(folds[0] + folds[1]),
        'boundary_train': sorted(folds[2]),
        'boundary_decision': sorted(folds[3]),
    }


def build_plan(data_root, seed=42, scenes=None):
    data_root = Path(data_root).resolve()
    scene_map = SCENES if scenes is None else scenes
    manifests = {scene: {split: [] for split in SPLITS} for scene in scene_map}
    summary = {
        'schema_version': 1,
        'strategy': 'per_subvideo_seeded_four_folds__fold01_probe__fold2_boundary_train__fold3_boundary_decision',
        'seed': int(seed),
        'ratios': {'probe_train': '1/2', 'boundary_train': '1/4', 'boundary_decision': '1/4'},
        'overlap': False,
        'data_root': str(data_root),
        'scenes': {},
    }

    for scene, folder in scene_map.items():
        scene_root = data_root / folder
        subvideos = sorted(path for path in scene_root.glob('sequence*/train/*') if path.is_dir())
        if not subvideos:
            raise FileNotFoundError(f'no subvideos found under {scene_root}/sequence*/train/*')
        scene_info = {'folder': folder, 'total_frames': 0, 'split_counts': {s: 0 for s in SPLITS},
                      'subvideos': []}
        all_scene_frames = []
        for subvideo_path in subvideos:
            frames = sorted(subvideo_path.glob('*.jpg'))
            if not frames:
                raise FileNotFoundError(f'no .jpg frames in subvideo {subvideo_path}')
            relative_subvideo = subvideo_path.relative_to(data_root).as_posix()
            relative_frames = [path.relative_to(data_root).as_posix() for path in frames]
            split_seed = stable_subvideo_seed(seed, scene, relative_subvideo)
            selected = split_subvideo(relative_frames, split_seed)
            counts = {name: len(selected[name]) for name in SPLITS}
            for split in SPLITS:
                manifests[scene][split].extend(selected[split])
                scene_info['split_counts'][split] += counts[split]
            scene_info['subvideos'].append({
                'path': relative_subvideo,
                'frames': len(frames),
                'seed': split_seed,
                'split_counts': counts,
            })
            scene_info['total_frames'] += len(frames)
            all_scene_frames.extend(relative_frames)

        for split in SPLITS:
            manifests[scene][split].sort()
        assert_partition(scene, all_scene_frames, manifests[scene], scene_info['subvideos'])
        scene_info['dataset_paths_sha256'] = sha256_bytes(
            ('\n'.join(sorted(all_scene_frames)) + '\n').encode('utf-8'))
        summary['scenes'][scene] = scene_info

    return manifests, summary


def assert_partition(scene, all_frames, split_frames, subvideos=None):
    expected = set(all_frames)
    actual_sets = {name: set(split_frames[name]) for name in SPLITS}
    for name in SPLITS:
        if len(actual_sets[name]) != len(split_frames[name]):
            raise ValueError(f'{scene}/{name}: duplicate frame in manifest')
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = actual_sets[left] & actual_sets[right]
            if overlap:
                raise ValueError(f'{scene}: {left} and {right} overlap by {len(overlap)} frames')
    union = set().union(*actual_sets.values())
    if union != expected:
        raise ValueError(
            f'{scene}: partition coverage mismatch missing={len(expected - union)} extra={len(union - expected)}')
    if subvideos:
        for item in subvideos:
            counts = item['split_counts']
            if sum(counts.values()) != item['frames']:
                raise ValueError(f"{scene}/{item['path']}: split counts do not cover subvideo")
            quarters = quarter_sizes(item['frames'])
            expected_counts = {
                'probe_train': quarters[0] + quarters[1],
                'boundary_train': quarters[2],
                'boundary_decision': quarters[3],
            }
            if counts != expected_counts:
                raise ValueError(
                    f"{scene}/{item['path']}: expected {expected_counts}, found {counts}")


def manifest_content(paths):
    return ('\n'.join(paths) + '\n').encode('utf-8')


def materialized_summary(manifests, summary):
    result = json.loads(json.dumps(summary))
    result['manifests'] = {}
    combined = hashlib.sha256()
    for scene in summary['scenes']:
        result['manifests'][scene] = {}
        for split in SPLITS:
            relative_path = f'{scene}/{split}.txt'
            content = manifest_content(manifests[scene][split])
            digest = sha256_bytes(content)
            combined.update(scene.encode('utf-8'))
            combined.update(split.encode('utf-8'))
            combined.update(bytes.fromhex(digest))
            result['manifests'][scene][split] = {
                'path': relative_path,
                'frames': len(manifests[scene][split]),
                'sha256': digest,
            }
    result['split_fingerprint_sha256'] = combined.hexdigest()
    return result


def write_or_verify(output_root, manifests, summary, write):
    output_root = Path(output_root)
    expected_summary = materialized_summary(manifests, summary)
    differences = []
    for scene in SCENES:
        for split in SPLITS:
            path = output_root / scene / f'{split}.txt'
            expected = manifest_content(manifests[scene][split])
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expected)
            elif not path.is_file():
                differences.append(f'missing {path}')
            elif path.read_bytes() != expected:
                differences.append(f'content mismatch {path}')

    summary_path = output_root / 'split_summary.json'
    expected_json = (json.dumps(expected_summary, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if write:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_bytes(expected_json)
    elif not summary_path.is_file():
        differences.append(f'missing {summary_path}')
    elif summary_path.read_bytes() != expected_json:
        differences.append(f'content mismatch {summary_path}')
    if differences:
        raise RuntimeError('fixed split audit failed:\n  ' + '\n  '.join(differences))
    return expected_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiment_mask_DLBM' / 'splits' / 'seed42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check', action='store_true', help='audit existing manifests without writing')
    args = parser.parse_args()

    manifests, summary = build_plan(args.data_root, seed=args.seed)
    result = write_or_verify(args.output, manifests, summary, write=not args.check)
    mode = 'verified' if args.check else 'written and verified'
    print(f"fixed splits {mode}: {args.output}")
    print(f"split fingerprint: {result['split_fingerprint_sha256']}")
    for scene in SCENES:
        counts = result['scenes'][scene]['split_counts']
        print(f"  {scene:10s} total={result['scenes'][scene]['total_frames']:5d} "
              f"probe={counts['probe_train']:5d} boundary_train={counts['boundary_train']:5d} "
              f"boundary_decision={counts['boundary_decision']:5d}")


if __name__ == '__main__':
    main()
