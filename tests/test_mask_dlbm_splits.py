from experiment_mask_DLBM.prepare_splits import quarter_sizes, split_subvideo


def test_quarter_sizes_cover_frames_and_remain_balanced():
    for length in range(1, 50):
        sizes = quarter_sizes(length)
        assert sum(sizes) == length
        assert max(sizes) - min(sizes) <= 1


def test_subvideo_split_is_reproducible_disjoint_and_complete():
    frames = [f'Scene/sequence1/train/01/{index:06d}.jpg' for index in range(37)]
    first = split_subvideo(frames, seed=12345)
    second = split_subvideo(frames, seed=12345)
    different = split_subvideo(frames, seed=54321)
    assert first == second
    assert first != different
    sets = {name: set(paths) for name, paths in first.items()}
    assert sets['probe_train'].isdisjoint(sets['boundary_train'])
    assert sets['probe_train'].isdisjoint(sets['boundary_decision'])
    assert sets['boundary_train'].isdisjoint(sets['boundary_decision'])
    assert set().union(*sets.values()) == set(frames)
    assert [len(first[name]) for name in first] == [19, 9, 9]
