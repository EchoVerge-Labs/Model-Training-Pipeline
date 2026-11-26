# Variant overlays

A variant holds only the keys it changes; it is deep-merged over the model's
`params-<model>.yaml` (and `params-data.yaml`). Pass one or more with
`VARIANT=configs/variants/cnn.yaml,configs/variants/homog.yaml`, and print the
resulting run name with `make name MODEL=<key> VARIANT=...`.

| File | Changes | Name token |
|---|---|---|
| `cnn.yaml` | the CNN feature encoder trains | `-cnn` |
| `homog.yaml` | language-homogeneous updates | `-homog` |
| `adapter.yaml` / `lora.yaml` / `gclora.yaml` | PE-CoPT method (templates, not implemented yet) | method slot |
| `smoke-1h.yaml` | 1 h selection, 30 passes (68 updates), ER-only benchmark | `-si-ta-1h-smoke` |

A variant may change any setting, but one the name does not show (anything other than
the CNN switch, the schedule length, the method, the hours and language batching) needs a `tag`, or two different runs would share one name.
