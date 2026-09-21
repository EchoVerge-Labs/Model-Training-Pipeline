# Data Selection Report

Target: 200.0h | Realised: 193.6h
Holdout: 2.1h
Longest training clip: 811.8s

## Catalog cleaning

- to_train=yes rows: 33,307
- exact repeats removed: 996
- unresolvable, set aside (see reports/unresolved_clips.csv): 23 (ambiguous 19, no_match 4, duplicate_path 0)
- candidates for selection: 32,288

## By Language

- sinhala: 94.5h
- tamil: 99.2h

## By Genre (top 10)

- education: 39.6h
- news: 34.7h
- interview: 29.6h
- podcast_or_discussion: 28.0h
- drama: 26.3h
- political_speech: 23.7h
- vlog: 11.7h

## Top 10 sources by hours

Check for a single source dominating the training set.

| Rank | source_id | Hours | % of train | Segments | Title |
|---|---|---|---|---|---|
| 1 | 403fad9557ad1996 | 0.99 | 0.5% | 138 | 04 වන පාඩම - ශ්‍රී ලංකාවේ පැරණි සමාජය - 01 වන කොටස \| Grade 10 \| History Unit 4 Part 01 |
| 2 | 416e0fbe-b75d-40b6-a87e-a60034f39d9a | 0.98 | 0.5% | 55 | සිංහල Podcast \| How to prioritize you? |
| 3 | e7612bfe-dc1b-4520-a688-743b235234a8 | 0.85 | 0.4% | 13 | සිංහල Podcast \| How to improve a Grinding mindset |
| 4 | 87d670d9b33f5547 | 0.83 | 0.4% | 142 | 04 වන පාඩම - ශ්‍රී ලංකාවේ පැරණි සමාජය - 02 වන කොටස \| Grade 10 \| History Unit 4 Part 02 |
| 5 | 9fd3603eb2c7c57a | 0.82 | 0.4% | 80 | බුදු පිළිම විකුනලා සෙනඟ හොයන නාමල්- Politicore Podcast 152 |
| 6 | 4dbd1a5d-520f-4605-a0ca-39f930b2f0c3 | 0.81 | 0.4% | 82 | ඡන්ද ගේම් හොඳට බලලා මේ දේවල් ඔලුවටම දාගමු \| Podcast Epi 05 |
| 7 | 9ed21a874cc39c75 | 0.79 | 0.4% | 141 | 04 වන පාඩම - ශ්‍රී ලංකාවේ පැරණි සමාජය - 03 වන කොටස \| Grade 10 \| History Unit 4 Part 03 |
| 8 | 692db4d7-f772-46b1-adb5-bcd716425329 | 0.75 | 0.4% | 98 | දිලිප් චමත් මාරු ඩබලක් එක්ක මරු කයියක් \| Av Rase Podcast 30 \| JPG වීඩියෝ බලන්න 😂🤣 |
| 9 | 3490cdea-7af5-4173-bd10-1e51ac9bbd1b | 0.70 | 0.4% | 41 | සිංහල Podcast \| How to find you |
| 10 | fca469f7-924b-4abe-9c47-87befe9ed5d0 | 0.65 | 0.3% | 83 | My 10 biggest mistakes \|Podcast with Harsha Rathnayake \| Business Advisor |

## Channels: 4974 train, 52 holdout, 0 overlap
