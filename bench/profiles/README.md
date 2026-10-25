# Surveyed expert-usage profiles

`mimo26_decode.usage`, `mimo26_prefill.usage` and `mimo26_combined.usage` (decode + prefill counts) are MiMo V2.6 Flash
route counts, "layer expert count" lines, recorded 2026-10-23 over 20 varied prompts with
`bench/mimo_survey.sh survey` (graph_route_survey.hpp, decode and prefill counted separately).
`mimo26_combined.usage` is the one the configs use: against the earlier profile it gave about +8% decode on held-out
prompts at equal short-prompt prefill (docs/models/mimo26-flash.md). Regenerate after a change of model quant or
routing; the repo's `.gitignore` excludes `.zeromoe_usage*`, which is why these live here.
