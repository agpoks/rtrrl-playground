# Papers

The full list, with why each one is here, is in
[`papers/README.md`](https://github.com/agpoks/rtrrl-playground/blob/main/papers/README.md);
BibTeX for all of them is in `papers/references.bib`. The short version:

## The algorithm

* **Lemmel & Grosu, *Real-Time Recurrent Reinforcement Learning*, AAAI 2025**
  ([arXiv:2311.04830](https://arxiv.org/abs/2311.04830)) -- what this repo
  implements.
* **Lemmel, Resch, Farsang, Hasani, Rus & Grosu, *Online Fine-Tuning of
  Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL*,
  2026** ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)) -- RTRRL on a
  real 1:10 RoboRacer with an event camera. Two findings from it shape this
  repo: the LRC cell is the one that gets on best with RTRRL, and the natural
  deployment is fine-tuning a pretrained controller rather than learning from
  scratch.
* **Lemmel & Grosu, *On the Benefits of Biophysical Synapses*, AAAI 2023**
  ([arXiv:2303.04944](https://arxiv.org/abs/2303.04944)) -- the per-synapse
  nonlinearity that this repo's cells deliberately leave out, and the reference
  for what is being left out.

## The cells

Hasani et al. 2021 (LTC); Farsang, Neubauer & Grosu 2024 (LRC); Funahashi &
Nakamura 1993 (CT-RNN); Ravanelli et al. 2018 (LiGRU). See {doc}`cells`.

## The online gradient

Williams & Zipser 1989 (RTRL); Murray 2019 (RFLO); Tallec & Ollivier 2018
(UORO); Menick et al. 2021 (SnAp); Lillicrap et al. 2016 (feedback alignment);
Marschall, Cho & Savin 2020 (the unified framework that maps all of them). See
{doc}`estimators`.

**e-prop (Bellec et al. 2020) is not a separate option here**, on purpose. Its
eligibility trace *is* RFLO's influence for a leaky unit, combined with a
learning signal from outside; RTRRL is, in that framing, e-prop with a TD(λ)
actor-critic supplying the signal. A fifth flag would have produced a
near-duplicate of `rflo` and hidden that.

## The learning rule

Sutton 1988 (TD(λ)); van Seijen et al. 2016 (true online TD(λ), the Dutch
trace); Mnih et al. 2016 (A2C, the BPTT baseline).
