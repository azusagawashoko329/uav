"""DLBM v001: relation-assisted masked distance prediction and an ordered change point.

The frozen Probe supplies twelve Teacher/Student patch pairs.  This module:

1. keeps the Probe cosine-distance map as the explicit, task-aligned observation;
2. lets a small relation query read Teacher, Student, and distance at every patch;
3. fuses that learned relation only as a zero-initialized residual over a 3x3 distance stem;
4. predicts every shallower distance map from the already observed deeper suffix;
5. encodes the complete before/predicted/observed/after opening transition; and
6. fits one ordered two-regime density whose latent change point is the suffix boundary.

No train/decision discrepancy, anomaly label, Gate label, D/M/S group, threshold, or AUC target is
used.  L11 is the fixed inside-suffix anchor, and every legal decision is exactly {B, ..., L11}.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_module(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ConvGRUCell(nn.Module):
    """Spatial recurrent update shared by all twelve layer openings."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, value, hidden):
        if value.shape != hidden.shape:
            raise ValueError(
                f'ConvGRU value/hidden mismatch: {tuple(value.shape)} vs {tuple(hidden.shape)}')
        reset, update = self.gates(torch.cat((value, hidden), dim=1)).chunk(2, dim=1)
        reset, update = reset.sigmoid(), update.sigmoid()
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class PatchRelationEncoder(nn.Module):
    """Produce one learned relation token from T/S/distance at the same patch.

    The learned query is the only retained output.  Teacher and Student share the same projection
    because the Probe trained them in a common feature coordinate system.
    """

    def __init__(self, feature_dim=768, token_dim=64, heads=4):
        super().__init__()
        if token_dim % heads:
            raise ValueError('token_dim must be divisible by heads')
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.shared_projection = nn.Linear(feature_dim, token_dim, bias=False)
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim),
        )
        self.modality_embedding = nn.Parameter(torch.empty(4, token_dim))
        self.relation_query = nn.Parameter(torch.empty(1, 1, token_dim))
        self.query_norm = nn.LayerNorm(token_dim)
        self.value_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim, heads, dropout=0.0, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim), nn.GELU(),
            nn.Linear(4 * token_dim, token_dim),
        )
        self.global_query = nn.Parameter(torch.empty(1, 1, token_dim))
        self.global_norm = nn.LayerNorm(token_dim)
        self.global_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=0.0, batch_first=True,
        )
        self.global_ffn_norm = nn.LayerNorm(token_dim)
        self.global_ffn = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim), nn.GELU(),
            nn.Linear(2 * token_dim, token_dim),
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.relation_query, std=0.02)
        nn.init.normal_(self.global_query, std=0.02)

    def forward(self, teacher, student, distance):
        if teacher.shape != student.shape or teacher.ndim != 3:
            raise ValueError('expected matched Teacher/Student tensors shaped (B,P,C)')
        if teacher.shape[-1] != self.feature_dim:
            raise ValueError(
                f'expected feature_dim={self.feature_dim}, got {teacher.shape[-1]}')
        if distance.shape != teacher.shape[:2]:
            raise ValueError(
                f'distance shape {tuple(distance.shape)} does not match {tuple(teacher.shape[:2])}')

        # Cosine RD is magnitude invariant, so relation projection receives unit feature directions.
        teacher = F.normalize(teacher.float(), dim=-1)
        student = F.normalize(student.float(), dim=-1)
        teacher_token = self.shared_projection(teacher) + self.modality_embedding[0]
        student_token = self.shared_projection(student) + self.modality_embedding[1]
        distance_token = (
            self.distance_embedding(distance.float().unsqueeze(-1))
            + self.modality_embedding[2]
        )
        values = torch.stack((teacher_token, student_token, distance_token), dim=-2)
        flat_values = values.reshape(-1, 3, self.token_dim)
        query = (
            self.relation_query + self.modality_embedding[3].view(1, 1, -1)
        ).expand(len(flat_values), -1, -1)
        attended, _ = self.attention(
            self.query_norm(query), self.value_norm(flat_values),
            self.value_norm(flat_values), need_weights=False,
        )
        relation = query + attended
        relation = relation + self.ffn(self.output_norm(relation))
        relation = relation[:, 0].reshape(
            teacher.shape[0], teacher.shape[1], self.token_dim,
        )

        global_query = self.global_query.expand(len(teacher), -1, -1)
        global_value, _ = self.global_attention(
            self.global_norm(global_query), self.global_norm(relation),
            self.global_norm(relation), need_weights=False,
        )
        global_value = global_query + global_value
        global_value = global_value + self.global_ffn(
            self.global_ffn_norm(global_value),
        )
        return relation, global_value[:, 0]


class MaskedRelationDLBM(nn.Module):
    """Deep-to-shallow conditional predictor and transition-token encoder."""

    def __init__(
        self, feature_dim=768, num_layers=12, token_dim=64, state_dim=32, heads=4,
        patch_count=256,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_layers = int(num_layers)
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.patch_count = int(patch_count)
        self.side = math.isqrt(self.patch_count)
        if self.side * self.side != self.patch_count:
            raise ValueError('patch_count must form a square grid')
        if self.num_layers < 2:
            raise ValueError('at least two layers are required')

        self.relation_encoder = PatchRelationEncoder(
            feature_dim=feature_dim, token_dim=token_dim, heads=heads,
        )
        self.distance_stem = nn.Sequential(
            nn.Conv2d(1, state_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(state_dim, state_dim, 3, padding=1), nn.GELU(),
        )
        # Zero initialization makes the first optimization step exactly the validated D-map core.
        self.relation_to_state = _zero_module(nn.Conv2d(token_dim, state_dim, 1))
        self.global_to_state = _zero_module(nn.Linear(token_dim, state_dim))
        self.observed_layer_embedding = nn.Parameter(torch.empty(num_layers, state_dim))
        self.target_layer_embedding = nn.Parameter(torch.empty(num_layers, state_dim))
        self.recurrent = ConvGRUCell(state_dim)
        self.distance_decoder = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(state_dim, 1, 1),
        )

        transition_input_dim = 3 * state_dim + 2  # before, observed, after, predicted D, actual D
        self.transition_stem = nn.Sequential(
            nn.Conv2d(transition_input_dim, token_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(token_dim, token_dim, 3, padding=1), nn.GELU(),
        )
        self.transition_query = nn.Parameter(torch.empty(1, 1, token_dim))
        self.transition_norm = nn.LayerNorm(token_dim)
        self.transition_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=0.0, batch_first=True,
        )
        self.transition_local_norm = nn.LayerNorm(token_dim)
        self.transition_relation = nn.Linear(token_dim, token_dim)
        self.transition_output_norm = nn.LayerNorm(token_dim)
        self.transition_decoder = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim), nn.GELU(),
            nn.Linear(4 * token_dim, patch_count),
        )
        nn.init.normal_(self.observed_layer_embedding, std=0.02)
        nn.init.normal_(self.target_layer_embedding, std=0.02)
        nn.init.normal_(self.transition_query, std=0.02)

    @staticmethod
    def cosine_distance(teacher, student):
        return 1.0 - F.cosine_similarity(teacher.float(), student.float(), dim=-1)

    def _validate_inputs(self, teacher_taps, student_taps):
        if len(teacher_taps) != self.num_layers or len(student_taps) != self.num_layers:
            raise ValueError(
                f'expected {self.num_layers} Teacher/Student taps, got '
                f'{len(teacher_taps)}/{len(student_taps)}')
        reference = teacher_taps[0].shape
        expected = (reference[0], self.patch_count, self.feature_dim)
        if tuple(reference) != expected:
            raise ValueError(f'expected tap shape {expected}, got {tuple(reference)}')
        for layer, (teacher, student) in enumerate(zip(teacher_taps, student_taps)):
            if teacher.shape != reference or student.shape != reference:
                raise ValueError(f'inconsistent tap shapes at layer {layer}')
            if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
                raise FloatingPointError(f'non-finite Probe feature at layer {layer}')

    def _encode_observations(self, teacher_taps, student_taps, use_relation):
        relations, globals_, distances = [], [], []
        for teacher, student in zip(teacher_taps, student_taps):
            distance = self.cosine_distance(teacher, student)
            distances.append(distance)
            if use_relation:
                relation, global_token = self.relation_encoder(
                    teacher, student, distance,
                )
            else:
                relation = distance.new_zeros(
                    (len(distance), self.patch_count, self.token_dim),
                )
                global_token = distance.new_zeros((len(distance), self.token_dim))
            relations.append(relation)
            globals_.append(global_token)
        distance = torch.stack(distances, dim=1)
        relation = torch.stack(relations, dim=1)
        global_token = torch.stack(globals_, dim=1)

        flat_distance = distance.reshape(
            -1, 1, self.side, self.side,
        )
        distance_state = self.distance_stem(flat_distance)
        relation_grid = relation.permute(0, 1, 3, 2).reshape(
            -1, self.token_dim, self.side, self.side,
        )
        relation_state = self.relation_to_state(relation_grid)
        global_state = self.global_to_state(
            global_token.reshape(-1, self.token_dim),
        ).view(-1, self.state_dim, 1, 1)
        observation = distance_state + relation_state + global_state
        observation = observation.reshape(
            len(distance), self.num_layers, self.state_dim, self.side, self.side,
        )
        distance_state = distance_state.reshape_as(observation)
        residual_ratio = (
            (relation_state + global_state).detach().abs().mean()
            / distance_state.detach().abs().mean().clamp_min(1e-8)
        )
        return distance, observation, global_token, residual_ratio

    def _transition_token(
        self, before, observed, after, predicted, actual, relation_global,
    ):
        transition = self.transition_stem(torch.cat((
            before, observed, after,
            predicted.unsqueeze(1), actual.unsqueeze(1),
        ), dim=1))
        local = self.transition_local_norm(transition.flatten(2).transpose(1, 2))
        query = self.transition_query.expand(len(local), -1, -1)
        attended, _ = self.transition_attention(
            self.transition_norm(query), self.transition_norm(local),
            self.transition_norm(local), need_weights=False,
        )
        token = query[:, 0] + attended[:, 0] + self.transition_relation(relation_global)
        return self.transition_output_norm(token), local

    def forward(self, teacher_taps, student_taps, use_relation=True):
        self._validate_inputs(teacher_taps, student_taps)
        distances, observations, relation_globals, residual_ratio = self._encode_observations(
            teacher_taps, student_taps, use_relation=bool(use_relation),
        )
        batch = len(distances)
        hidden = distances.new_zeros(
            (batch, self.state_dim, self.side, self.side),
        )
        transition_tokens = [None] * self.num_layers
        transition_locals = [None] * self.num_layers
        transition_targets = [None] * self.num_layers
        predictions = [None] * (self.num_layers - 1)

        # L11 is always observed and is the fixed inside-suffix anchor.
        layer = self.num_layers - 1
        before = hidden
        observed = observations[:, layer] + self.observed_layer_embedding[layer].view(
            1, -1, 1, 1,
        )
        after = self.recurrent(observed, before)
        anchor_actual = distances[:, layer].reshape(batch, self.side, self.side)
        anchor_prediction = anchor_actual.new_zeros(anchor_actual.shape)
        transition_tokens[layer], transition_locals[layer] = self._transition_token(
            before, observed, after, anchor_prediction, anchor_actual,
            relation_globals[:, layer],
        )
        transition_targets[layer] = distances[:, layer]
        hidden = after

        # Before Lk is observed, only H_{k+1} from the already-open deeper suffix predicts Dk.
        for layer in range(self.num_layers - 2, -1, -1):
            before = hidden
            predicted = self.distance_decoder(
                before + self.target_layer_embedding[layer].view(1, -1, 1, 1),
            ).squeeze(1)
            actual = distances[:, layer].reshape(batch, self.side, self.side)
            observed = observations[:, layer] + self.observed_layer_embedding[layer].view(
                1, -1, 1, 1,
            )
            after = self.recurrent(observed, before)
            predictions[layer] = predicted.flatten(1)
            # The transition decoder must encode the realized innovation; its target may not move
            # with the predictor to make the auxiliary task artificially easier.
            transition_targets[layer] = (actual - predicted.detach()).flatten(1)
            transition_tokens[layer], transition_locals[layer] = self._transition_token(
                before, observed, after, predicted, actual, relation_globals[:, layer],
            )
            hidden = after

        prediction = torch.stack(predictions, dim=1)
        transition_token = torch.stack(transition_tokens, dim=1)
        transition_local = torch.stack(transition_locals, dim=1)
        # One learned whole-layer token plus all 256 spatial transition tokens.
        transition_set = torch.cat(
            (transition_token.unsqueeze(2), transition_local), dim=2,
        )
        transition_target = torch.stack(transition_targets, dim=1)
        transition_reconstruction = self.transition_decoder(transition_token)
        if not all(torch.isfinite(value).all() for value in (
            prediction, transition_token, transition_set, transition_reconstruction,
        )):
            raise FloatingPointError('non-finite DLBM output')
        return {
            'distance': distances,
            'prediction': prediction,
            'prediction_target': distances[:, :-1],
            'transition_token': transition_token,
            'transition_set': transition_set,
            'transition_reconstruction': transition_reconstruction,
            'transition_target': transition_target,
            'relation_residual_ratio': residual_ratio,
        }


def representation_loss(output):
    prediction = F.smooth_l1_loss(
        output['prediction'], output['prediction_target'],
    )
    transition = F.smooth_l1_loss(
        output['transition_reconstruction'], output['transition_target'],
    )
    total = 0.5 * (prediction + transition)
    return total, {'prediction': prediction, 'transition': transition}


class OrderedChangePointDensity(nn.Module):
    """A proper two-regime diagonal Gaussian density over frozen transition tokens."""

    OUTSIDE = 0
    INSIDE = 1

    def __init__(self, token_dim=64, num_layers=12, min_scale=0.05):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_layers = int(num_layers)
        self.min_scale = float(min_scale)
        self.means = nn.Parameter(torch.zeros(2, token_dim))
        self.raw_scales = nn.Parameter(torch.zeros(2, token_dim))

    def _validate_tokens(self, tokens):
        valid = (
            tokens.ndim == 3
            and tokens.shape[1:] == (self.num_layers, self.token_dim)
        ) or (
            tokens.ndim == 4
            and tokens.shape[1] == self.num_layers
            and tokens.shape[-1] == self.token_dim
            and tokens.shape[2] >= 2
        )
        if not valid:
            raise ValueError(
                f'expected (B,L,D) or (B,L,N,D) transition tokens, got {tuple(tokens.shape)}')

    @staticmethod
    def _endpoint_moments(values):
        values = values.float()
        if values.ndim == 2:
            return values.mean(dim=0), values.square().mean(dim=0)
        # Give the one global token and the complete local field equal scale-level weight.
        global_value = values[:, 0]
        local_value = values[:, 1:]
        mean = 0.5 * global_value.mean(dim=0) + 0.5 * local_value.mean(dim=(0, 1))
        second = (
            0.5 * global_value.square().mean(dim=0)
            + 0.5 * local_value.square().mean(dim=(0, 1))
        )
        return mean, second

    def initialize_from_endpoints(self, tokens):
        self._validate_tokens(tokens)
        if tokens.shape[1] != self.num_layers:
            raise ValueError('invalid transition-token cache for density initialization')
        with torch.no_grad():
            endpoints = (tokens[:, 0], tokens[:, -1])
            for mode, values in zip((self.OUTSIDE, self.INSIDE), endpoints):
                mean, second = self._endpoint_moments(values)
                scale = (second - mean.square()).clamp_min(0).sqrt().clamp_min(self.min_scale)
                self.means[mode].copy_(mean)
                inverse_softplus = torch.log(torch.expm1(
                    (scale - self.min_scale).clamp_min(1e-4),
                ))
                self.raw_scales[mode].copy_(inverse_softplus)

    def initialize_from_moments(self, moments):
        required = ('global_mean', 'global_second', 'local_mean', 'local_second')
        if any(key not in moments for key in required):
            raise ValueError('incomplete transition-set moments')
        with torch.no_grad():
            for mode, layer in ((self.OUTSIDE, 0), (self.INSIDE, self.num_layers - 1)):
                mean = (
                    0.5 * moments['global_mean'][layer]
                    + 0.5 * moments['local_mean'][layer]
                )
                second = (
                    0.5 * moments['global_second'][layer]
                    + 0.5 * moments['local_second'][layer]
                )
                scale = (second - mean.square()).clamp_min(0).sqrt().clamp_min(self.min_scale)
                self.means[mode].copy_(mean)
                inverse_softplus = torch.log(torch.expm1(
                    (scale - self.min_scale).clamp_min(1e-4),
                ))
                self.raw_scales[mode].copy_(inverse_softplus)

    def scales(self):
        return F.softplus(self.raw_scales) + self.min_scale

    def emission_log_prob(self, tokens):
        self._validate_tokens(tokens)
        mode_axis = 2 if tokens.ndim == 3 else 3
        difference = tokens.unsqueeze(mode_axis) - self.means.view(
            *((1,) * mode_axis), 2, self.token_dim,
        )
        scale = self.scales().view(*((1,) * mode_axis), 2, self.token_dim)
        log_prob = -0.5 * (
            (difference / scale).square() + 2.0 * scale.log() + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.mean(dim=-1)  # comparable across token dimensions
        if tokens.ndim == 4:
            return 0.5 * log_prob[:, :, 0] + 0.5 * log_prob[:, :, 1:].mean(dim=2)
        return log_prob

    def boundary_logits_from_emission(self, emission):
        if emission.ndim != 3 or emission.shape[1:] != (self.num_layers, 2):
            raise ValueError('invalid emission log-probability tensor')
        logits = []
        for boundary in range(self.num_layers):
            outside = emission[:, :boundary, self.OUTSIDE].sum(dim=1)
            inside = emission[:, boundary:, self.INSIDE].sum(dim=1)
            logits.append((outside + inside) / self.num_layers)
        return torch.stack(logits, dim=1)

    def boundary_logits(self, tokens):
        return self.boundary_logits_from_emission(self.emission_log_prob(tokens))

    def emission_log_prob_from_moments(self, moments):
        required = ('global_mean', 'global_second', 'local_mean', 'local_second')
        if any(key not in moments for key in required):
            raise ValueError('incomplete transition-set moments')
        mean = self.means.view(1, 2, self.token_dim)
        scale = self.scales().view(1, 2, self.token_dim)

        def expected_log_prob(data_mean, data_second):
            expected_square = (
                data_second.unsqueeze(1)
                - 2.0 * data_mean.unsqueeze(1) * mean
                + mean.square()
            ).clamp_min(0)
            return -0.5 * (
                expected_square / scale.square()
                + 2.0 * scale.log()
                + math.log(2.0 * math.pi)
            ).mean(dim=-1)

        return (
            0.5 * expected_log_prob(moments['global_mean'], moments['global_second'])
            + 0.5 * expected_log_prob(moments['local_mean'], moments['local_second'])
        )

    def scene_marginal_nll(self, tokens):
        # One scene has one B: aggregate frame evidence before marginalizing B.
        logits = self.boundary_logits(tokens).mean(dim=0)
        log_prior = -math.log(self.num_layers)
        return -(torch.logsumexp(logits + log_prior, dim=0)), logits

    def scene_marginal_nll_from_moments(self, moments):
        emission = self.emission_log_prob_from_moments(moments)
        logits = self.boundary_logits_from_emission(emission.unsqueeze(0))[0]
        log_prior = -math.log(self.num_layers)
        return -(torch.logsumexp(logits + log_prior, dim=0)), logits

    def forward(self, tokens):
        return self.boundary_logits(tokens)


def policy_from_scene_logits(logits):
    if logits.ndim != 1:
        raise ValueError('expected one scene logit per boundary')
    probabilities = logits.softmax(dim=0)
    boundary = int(probabilities.argmax())
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    normalized_entropy = float(entropy / math.log(len(probabilities)))
    indices = torch.arange(len(probabilities), device=probabilities.device)
    return {
        'boundary': boundary,
        'selected_taps': list(range(boundary, len(probabilities))),
        'probabilities': [float(value) for value in probabilities.detach().cpu()],
        'expected_boundary': float((probabilities * indices).sum()),
        'confidence': float(probabilities.max()),
        'normalized_entropy': normalized_entropy,
    }
