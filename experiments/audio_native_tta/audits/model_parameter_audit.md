# SSL-AASIST model parameter audit

- Model: `ssl_aasist_source`
- Baseline: `source-20260919T083421Z-a61b23:epoch_0007`
- Task weights: `trained_in_project`
- Parameter counts: `{'backend': 446920, 'frontend': 317390592, 'head': 322}`

## Normalization inventory

- BatchNorm1d: 6
- BatchNorm2d: 14
- LayerNorm: 57
- GroupNorm: 0
- other_affine_modulation: 21

## Pre-registered scopes

### `backend_norm_affine_v1` (2050 parameters, 40 tensors)
- `GAT_layer_S.bn.bias`
- `GAT_layer_S.bn.weight`
- `GAT_layer_T.bn.bias`
- `GAT_layer_T.bn.weight`
- `HtrgGAT_layer_ST11.bn.bias`
- `HtrgGAT_layer_ST11.bn.weight`
- `HtrgGAT_layer_ST12.bn.bias`
- `HtrgGAT_layer_ST12.bn.weight`
- `HtrgGAT_layer_ST21.bn.bias`
- `HtrgGAT_layer_ST21.bn.weight`
- `HtrgGAT_layer_ST22.bn.bias`
- `HtrgGAT_layer_ST22.bn.weight`
- `attention.2.bias`
- `attention.2.weight`
- `encoder.0.0.bn2.bias`
- `encoder.0.0.bn2.weight`
- `encoder.1.0.bn1.bias`
- `encoder.1.0.bn1.weight`
- `encoder.1.0.bn2.bias`
- `encoder.1.0.bn2.weight`
- `encoder.2.0.bn1.bias`
- `encoder.2.0.bn1.weight`
- `encoder.2.0.bn2.bias`
- `encoder.2.0.bn2.weight`
- `encoder.3.0.bn1.bias`
- `encoder.3.0.bn1.weight`
- `encoder.3.0.bn2.bias`
- `encoder.3.0.bn2.weight`
- `encoder.4.0.bn1.bias`
- `encoder.4.0.bn1.weight`
- `encoder.4.0.bn2.bias`
- `encoder.4.0.bn2.weight`
- `encoder.5.0.bn1.bias`
- `encoder.5.0.bn1.weight`
- `encoder.5.0.bn2.bias`
- `encoder.5.0.bn2.weight`
- `first_bn.bias`
- `first_bn.weight`
- `first_bn1.bias`
- `first_bn1.weight`

### `backend_norm_plus_graph_modulation_v1` (5506 parameters, 61 tensors)
- `GAT_layer_S.att_weight`
- `GAT_layer_S.bn.bias`
- `GAT_layer_S.bn.weight`
- `GAT_layer_T.att_weight`
- `GAT_layer_T.bn.bias`
- `GAT_layer_T.bn.weight`
- `HtrgGAT_layer_ST11.att_weight11`
- `HtrgGAT_layer_ST11.att_weight12`
- `HtrgGAT_layer_ST11.att_weight22`
- `HtrgGAT_layer_ST11.att_weightM`
- `HtrgGAT_layer_ST11.bn.bias`
- `HtrgGAT_layer_ST11.bn.weight`
- `HtrgGAT_layer_ST12.att_weight11`
- `HtrgGAT_layer_ST12.att_weight12`
- `HtrgGAT_layer_ST12.att_weight22`
- `HtrgGAT_layer_ST12.att_weightM`
- `HtrgGAT_layer_ST12.bn.bias`
- `HtrgGAT_layer_ST12.bn.weight`
- `HtrgGAT_layer_ST21.att_weight11`
- `HtrgGAT_layer_ST21.att_weight12`
- `HtrgGAT_layer_ST21.att_weight22`
- `HtrgGAT_layer_ST21.att_weightM`
- `HtrgGAT_layer_ST21.bn.bias`
- `HtrgGAT_layer_ST21.bn.weight`
- `HtrgGAT_layer_ST22.att_weight11`
- `HtrgGAT_layer_ST22.att_weight12`
- `HtrgGAT_layer_ST22.att_weight22`
- `HtrgGAT_layer_ST22.att_weightM`
- `HtrgGAT_layer_ST22.bn.bias`
- `HtrgGAT_layer_ST22.bn.weight`
- `attention.2.bias`
- `attention.2.weight`
- `encoder.0.0.bn2.bias`
- `encoder.0.0.bn2.weight`
- `encoder.1.0.bn1.bias`
- `encoder.1.0.bn1.weight`
- `encoder.1.0.bn2.bias`
- `encoder.1.0.bn2.weight`
- `encoder.2.0.bn1.bias`
- `encoder.2.0.bn1.weight`
- `encoder.2.0.bn2.bias`
- `encoder.2.0.bn2.weight`
- `encoder.3.0.bn1.bias`
- `encoder.3.0.bn1.weight`
- `encoder.3.0.bn2.bias`
- `encoder.3.0.bn2.weight`
- `encoder.4.0.bn1.bias`
- `encoder.4.0.bn1.weight`
- `encoder.4.0.bn2.bias`
- `encoder.4.0.bn2.weight`
- `encoder.5.0.bn1.bias`
- `encoder.5.0.bn1.weight`
- `encoder.5.0.bn2.bias`
- `encoder.5.0.bn2.weight`
- `first_bn.bias`
- `first_bn.weight`
- `first_bn1.bias`
- `first_bn1.weight`
- `master1`
- `master2`
- `pos_S`

## Candidate modules

| module | type | component | origin | parameter count | parameters |
|---|---|---|---|---:|---|
| `ssl_model.model.feature_extractor.conv_layers.0.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.0.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.0.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.1.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.1.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.1.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.2.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.2.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.2.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.3.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.3.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.3.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.4.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.4.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.4.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.5.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.5.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.5.2.1.bias` |
| `ssl_model.model.feature_extractor.conv_layers.6.2.1` | Fp32LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.feature_extractor.conv_layers.6.2.1.weight`, `ssl_model.model.feature_extractor.conv_layers.6.2.1.bias` |
| `ssl_model.model.encoder.layers.0.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.0.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.0.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.0.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.0.final_layer_norm.weight`, `ssl_model.model.encoder.layers.0.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.1.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.1.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.1.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.1.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.1.final_layer_norm.weight`, `ssl_model.model.encoder.layers.1.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.2.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.2.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.2.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.2.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.2.final_layer_norm.weight`, `ssl_model.model.encoder.layers.2.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.3.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.3.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.3.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.3.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.3.final_layer_norm.weight`, `ssl_model.model.encoder.layers.3.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.4.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.4.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.4.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.4.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.4.final_layer_norm.weight`, `ssl_model.model.encoder.layers.4.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.5.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.5.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.5.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.5.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.5.final_layer_norm.weight`, `ssl_model.model.encoder.layers.5.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.6.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.6.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.6.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.6.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.6.final_layer_norm.weight`, `ssl_model.model.encoder.layers.6.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.7.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.7.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.7.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.7.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.7.final_layer_norm.weight`, `ssl_model.model.encoder.layers.7.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.8.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.8.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.8.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.8.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.8.final_layer_norm.weight`, `ssl_model.model.encoder.layers.8.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.9.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.9.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.9.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.9.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.9.final_layer_norm.weight`, `ssl_model.model.encoder.layers.9.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.10.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.10.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.10.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.10.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.10.final_layer_norm.weight`, `ssl_model.model.encoder.layers.10.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.11.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.11.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.11.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.11.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.11.final_layer_norm.weight`, `ssl_model.model.encoder.layers.11.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.12.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.12.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.12.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.12.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.12.final_layer_norm.weight`, `ssl_model.model.encoder.layers.12.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.13.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.13.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.13.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.13.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.13.final_layer_norm.weight`, `ssl_model.model.encoder.layers.13.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.14.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.14.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.14.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.14.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.14.final_layer_norm.weight`, `ssl_model.model.encoder.layers.14.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.15.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.15.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.15.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.15.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.15.final_layer_norm.weight`, `ssl_model.model.encoder.layers.15.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.16.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.16.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.16.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.16.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.16.final_layer_norm.weight`, `ssl_model.model.encoder.layers.16.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.17.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.17.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.17.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.17.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.17.final_layer_norm.weight`, `ssl_model.model.encoder.layers.17.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.18.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.18.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.18.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.18.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.18.final_layer_norm.weight`, `ssl_model.model.encoder.layers.18.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.19.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.19.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.19.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.19.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.19.final_layer_norm.weight`, `ssl_model.model.encoder.layers.19.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.20.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.20.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.20.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.20.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.20.final_layer_norm.weight`, `ssl_model.model.encoder.layers.20.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.21.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.21.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.21.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.21.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.21.final_layer_norm.weight`, `ssl_model.model.encoder.layers.21.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.22.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.22.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.22.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.22.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.22.final_layer_norm.weight`, `ssl_model.model.encoder.layers.22.final_layer_norm.bias` |
| `ssl_model.model.encoder.layers.23.self_attn_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.23.self_attn_layer_norm.weight`, `ssl_model.model.encoder.layers.23.self_attn_layer_norm.bias` |
| `ssl_model.model.encoder.layers.23.final_layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layers.23.final_layer_norm.weight`, `ssl_model.model.encoder.layers.23.final_layer_norm.bias` |
| `ssl_model.model.encoder.layer_norm` | LayerNorm | frontend | generic-pretrained | 2048 | `ssl_model.model.encoder.layer_norm.weight`, `ssl_model.model.encoder.layer_norm.bias` |
| `ssl_model.model.layer_norm` | LayerNorm | frontend | generic-pretrained | 1024 | `ssl_model.model.layer_norm.weight`, `ssl_model.model.layer_norm.bias` |
| `first_bn` | BatchNorm2d | backend | task-trained | 2 | `first_bn.weight`, `first_bn.bias` |
| `first_bn1` | BatchNorm2d | backend | task-trained | 128 | `first_bn1.weight`, `first_bn1.bias` |
| `encoder.0.0.bn2` | BatchNorm2d | backend | task-trained | 64 | `encoder.0.0.bn2.weight`, `encoder.0.0.bn2.bias` |
| `encoder.1.0.bn1` | BatchNorm2d | backend | task-trained | 64 | `encoder.1.0.bn1.weight`, `encoder.1.0.bn1.bias` |
| `encoder.1.0.bn2` | BatchNorm2d | backend | task-trained | 64 | `encoder.1.0.bn2.weight`, `encoder.1.0.bn2.bias` |
| `encoder.2.0.bn1` | BatchNorm2d | backend | task-trained | 64 | `encoder.2.0.bn1.weight`, `encoder.2.0.bn1.bias` |
| `encoder.2.0.bn2` | BatchNorm2d | backend | task-trained | 128 | `encoder.2.0.bn2.weight`, `encoder.2.0.bn2.bias` |
| `encoder.3.0.bn1` | BatchNorm2d | backend | task-trained | 128 | `encoder.3.0.bn1.weight`, `encoder.3.0.bn1.bias` |
| `encoder.3.0.bn2` | BatchNorm2d | backend | task-trained | 128 | `encoder.3.0.bn2.weight`, `encoder.3.0.bn2.bias` |
| `encoder.4.0.bn1` | BatchNorm2d | backend | task-trained | 128 | `encoder.4.0.bn1.weight`, `encoder.4.0.bn1.bias` |
| `encoder.4.0.bn2` | BatchNorm2d | backend | task-trained | 128 | `encoder.4.0.bn2.weight`, `encoder.4.0.bn2.bias` |
| `encoder.5.0.bn1` | BatchNorm2d | backend | task-trained | 128 | `encoder.5.0.bn1.weight`, `encoder.5.0.bn1.bias` |
| `encoder.5.0.bn2` | BatchNorm2d | backend | task-trained | 128 | `encoder.5.0.bn2.weight`, `encoder.5.0.bn2.bias` |
| `attention.2` | BatchNorm2d | backend | task-trained | 256 | `attention.2.weight`, `attention.2.bias` |
| `GAT_layer_S.bn` | BatchNorm1d | backend | task-trained | 128 | `GAT_layer_S.bn.weight`, `GAT_layer_S.bn.bias` |
| `GAT_layer_T.bn` | BatchNorm1d | backend | task-trained | 128 | `GAT_layer_T.bn.weight`, `GAT_layer_T.bn.bias` |
| `HtrgGAT_layer_ST11.bn` | BatchNorm1d | backend | task-trained | 64 | `HtrgGAT_layer_ST11.bn.weight`, `HtrgGAT_layer_ST11.bn.bias` |
| `HtrgGAT_layer_ST12.bn` | BatchNorm1d | backend | task-trained | 64 | `HtrgGAT_layer_ST12.bn.weight`, `HtrgGAT_layer_ST12.bn.bias` |
| `HtrgGAT_layer_ST21.bn` | BatchNorm1d | backend | task-trained | 64 | `HtrgGAT_layer_ST21.bn.weight`, `HtrgGAT_layer_ST21.bn.bias` |
| `HtrgGAT_layer_ST22.bn` | BatchNorm1d | backend | task-trained | 64 | `HtrgGAT_layer_ST22.bn.weight`, `HtrgGAT_layer_ST22.bn.bias` |
| `pos_S` | StandaloneParameter | backend | task-trained | 2688 | `pos_S` |
| `master1` | StandaloneParameter | backend | task-trained | 64 | `master1` |
| `master2` | StandaloneParameter | backend | task-trained | 64 | `master2` |
| `GAT_layer_S.att_weight` | StandaloneParameter | backend | task-trained | 64 | `GAT_layer_S.att_weight` |
| `GAT_layer_T.att_weight` | StandaloneParameter | backend | task-trained | 64 | `GAT_layer_T.att_weight` |
| `HtrgGAT_layer_ST11.att_weight11` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST11.att_weight11` |
| `HtrgGAT_layer_ST11.att_weight22` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST11.att_weight22` |
| `HtrgGAT_layer_ST11.att_weight12` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST11.att_weight12` |
| `HtrgGAT_layer_ST11.att_weightM` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST11.att_weightM` |
| `HtrgGAT_layer_ST12.att_weight11` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST12.att_weight11` |
| `HtrgGAT_layer_ST12.att_weight22` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST12.att_weight22` |
| `HtrgGAT_layer_ST12.att_weight12` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST12.att_weight12` |
| `HtrgGAT_layer_ST12.att_weightM` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST12.att_weightM` |
| `HtrgGAT_layer_ST21.att_weight11` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST21.att_weight11` |
| `HtrgGAT_layer_ST21.att_weight22` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST21.att_weight22` |
| `HtrgGAT_layer_ST21.att_weight12` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST21.att_weight12` |
| `HtrgGAT_layer_ST21.att_weightM` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST21.att_weightM` |
| `HtrgGAT_layer_ST22.att_weight11` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST22.att_weight11` |
| `HtrgGAT_layer_ST22.att_weight22` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST22.att_weight22` |
| `HtrgGAT_layer_ST22.att_weight12` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST22.att_weight12` |
| `HtrgGAT_layer_ST22.att_weightM` | StandaloneParameter | backend | task-trained | 32 | `HtrgGAT_layer_ST22.att_weightM` |
